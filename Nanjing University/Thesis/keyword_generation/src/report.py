# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:25:23 2019

@author: houwenxin
"""

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

def brief_report(epoch, batch_idx, one2one_batch, loss_ml, decoder_log_probs, opt, logging):
    logging.info('======================  Epoch: %d, Batch Index: %d  =========================' % (epoch, batch_idx))

    logging.info('Loss=%.5f' % (np.mean(loss_ml)))
    sampled_size = 2
    logging.info('Printing predictions on %d sampled examples by greedy search' % sampled_size)

    src, _, trg, trg_target, trg_copy_target, src_ext, oov_lists = one2one_batch
    if torch.cuda.is_available():
        src = src.data.cpu().numpy()
        decoder_log_probs = decoder_log_probs.data.cpu().numpy()
        max_words_pred = decoder_log_probs.argmax(axis=-1)
        trg_target = trg_target.data.cpu().numpy()
        trg_copy_target = trg_copy_target.data.cpu().numpy()
    else:
        src = src.data.numpy()
        decoder_log_probs = decoder_log_probs.data.numpy()
        max_words_pred = decoder_log_probs.argmax(axis=-1)
        trg_target = trg_target.data.numpy()
        trg_copy_target = trg_copy_target.data.numpy()

    sampled_trg_idx = np.random.randint(low=0, high=len(trg) - 1, size=sampled_size)
    src = src[sampled_trg_idx]
    oov_lists = [oov_lists[i] for i in sampled_trg_idx]
    max_words_pred = [max_words_pred[i] for i in sampled_trg_idx]
    decoder_log_probs = decoder_log_probs[sampled_trg_idx]
    if not opt.copy_attention:
        trg_target = [trg_target[i] for i in
                      sampled_trg_idx]  # use the real target trg_loss (the starting <BOS> has been removed and contains oov ground-truth)
    else:
        trg_target = [trg_copy_target[i] for i in sampled_trg_idx]

    for i, (src_wi, pred_wi, trg_i, oov_i) in enumerate(
            zip(src, max_words_pred, trg_target, oov_lists)):
        nll_prob = -np.sum([decoder_log_probs[i][l][pred_wi[l]] for l in range(len(trg_i))])
        find_copy = np.any([x >= opt.vocab_size for x in src_wi])
        has_copy = np.any([x >= opt.vocab_size for x in trg_i])

        sentence_source = [opt.id2word[x] if x < opt.vocab_size else oov_i[x - opt.vocab_size] for x in
                           src_wi]
        sentence_pred = [opt.id2word[x] if x < opt.vocab_size else oov_i[x - opt.vocab_size] for x in
                         pred_wi]
        sentence_real = [opt.id2word[x] if x < opt.vocab_size else oov_i[x - opt.vocab_size] for x in
                         trg_i]

        # index() 函数用于从列表中找出某个值第一个匹配项的索引位置
        sentence_source = sentence_source[:sentence_source.index(
            '<pad>')] if '<pad>' in sentence_source else sentence_source
        sentence_pred = sentence_pred[
            :sentence_pred.index('<pad>')] if '<pad>' in sentence_pred else sentence_pred
        sentence_real = sentence_real[
            :sentence_real.index('<pad>')] if '<pad>' in sentence_real else sentence_real

        logging.info('==================================================')
        logging.info('Source: %s ' % (' '.join(sentence_source)))
        logging.info('\t\tPred : %s (%.4f)' % (' '.join(sentence_pred), nll_prob) + (
            ' [FIND COPY]' if find_copy else ''))
        logging.info('\t\tReal : %s ' % (' '.join(sentence_real)) + (
            ' [HAS COPY]' + str(trg_i) if has_copy else ''))
        
        
def plot_learning_curve_and_write_csv(scores, curve_names, checkpoint_names, title, ylim=None, save_path=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    title : Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    """
    # linspace用法：numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    train_sizes=np.linspace(1, len(scores[0]), len(scores[0]))
    plt.figure(dpi=500)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.grid()
    means   = {}
    stds    = {}

    # colors = "rgbcmykw"
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(curve_names)))

    for i, (name, score) in enumerate(zip(curve_names, scores)):
        # get the mean and std of score along the time step
        mean = np.asarray([np.mean(s) for s in score])
        means[name] = mean
        std  = np.asarray([np.std(s) for s in score])
        stds[name] = std

        if name.lower().startswith('training ml'):
            score_ = [np.asarray(s) / 20.0 for s in score]
            mean = np.asarray([np.mean(s) for s in score_])
            std  = np.asarray([np.std(s) for s in score_])

        plt.fill_between(train_sizes, mean - std,
                         mean + std, alpha=0.1,
                         color=colors[i])
        plt.plot(train_sizes, mean, 'o-', color=colors[i],
                 label=name)

    plt.legend(loc="best", prop={'size': 6})
    # plt.show()
    if save_path:
        plt.savefig(save_path + '.png', bbox_inches='tight')

        csv_lines = ['time, ' + ','.join(curve_names)]
        for t_id, t in enumerate(checkpoint_names):
            csv_line = t + ',' + ','.join([str(means[c_name][t_id]) for c_name in curve_names])
            csv_lines.append(csv_line)

        with open(save_path + '.csv', 'w') as result_csv:
            result_csv.write('\n'.join(csv_lines))

    plt.close()
    return plt

class Progbar(object):
    def __init__(self, logger, title, target, width=30, batch_size = None, total_examples = None, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.logger = logger
        self.title = title
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

        self.batch_size = batch_size
        self.last_batch = 0
        self.total_examples = total_examples
        self.start_time = time.time() - 0.00001
        self.last_time  = self.start_time
        self.report_delay = 10
        self.last_report  = self.start_time

    def update(self, current_epoch, current, values=[]):
        '''
        @param current: index of current step
        @param values: list of tuples (name, value_for_last_step).
        The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1

            epoch_info = '%s Epoch=%d -' % (self.title, current_epoch) if current_epoch else '%s -' % (self.title)

            barstr = epoch_info + '%%%dd/%%%dd' % (numdigits, numdigits, ) + ' (%.2f%%)['
            bar = barstr % (current, self.target, float(current)/float(self.target) * 100.0)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('.'*(prog_width-1))
                if current < self.target:
                    bar += '(-w-)'
                else:
                    bar += '(-v-)!!'
            bar += ('~' * (self.width-prog_width))
            bar += ']'
            # sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)

            # info = ''
            info = bar
            if current < self.target:
                info += ' - Run-time: %ds - ETA: %ds' % (now - self.start, eta)
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                # info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                if k == 'perplexity' or k == 'PPL':
                    info += ' - %s: %.4f' % (k, np.exp(self.sum_values[k][0] / max(1, self.sum_values[k][1])))
                else:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))

            # update progress stats
            '''
            current_time = time.time()
            elapsed = current_time - self.last_report
            if elapsed > self.report_delay:
                trained_word_count = self.batch_size * current  # only words in vocab & sampled
                new_trained_word_count = self.batch_size * (current - self.last_batch)  # only words in vocab & sampled

                info += " - new processed %d words, %.0f words/s" % (new_trained_word_count, new_trained_word_count / elapsed)
                self.last_time   = current_time
                self.last_report = current_time
                self.last_batch  = current
            '''

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            # sys.stdout.write(info)
            # sys.stdout.flush()

            self.logger.info(info)

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                # sys.stdout.write(info + "\n")
                self.logger.critical(info + "\n")
                print(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)

    def clear(self):
        self.sum_values = {}
        self.unique_values = []
        self.total_width = 0
        self.seen_so_far = 0
   
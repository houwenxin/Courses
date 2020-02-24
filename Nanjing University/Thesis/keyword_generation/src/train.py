# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:34:37 2019

@author: houwenxin
"""

import torch
import vocab
from evaluate import evaluate_beam_search

import os
import time
import copy
import numpy as np
from conf import init_opts, init_logging
from dataloader import load_vocab_and_data
from model import init_model
from optimizer import init_optimizer_criterion
from beam_search import SequenceGenerator

from report import brief_report, plot_learning_curve_and_write_csv, Progbar

def train(one2one_batch, model, optimizer, criterion, opt):
    """
    text_lists_one2one, text_one2one_lens, keywords_lists_one2one, 
    keywords_target_lists_one2one, keywords_oov_target_lists_one2one, text_oov_lists_one2one, oov_lists_one2one
    """
    # one2one_batch返回的东西：
    # text_lists_one2one, text_one2one_lens, keywords_lists_one2one, 
    # keywords_target_lists_one2one, keywords_oov_target_lists_one2one, text_oov_lists_one2one, oov_lists_one2one
    src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists = one2one_batch
    max_oov_number = max([len(oov) for oov in oov_lists])

    # print("src size - ", src.size())
    # print("target size - ", trg.size())

    if torch.cuda.is_available():
        src = src.cuda()
        trg = trg.cuda()
        trg_target = trg_target.cuda()
        trg_copy_target = trg_copy_target.cuda()
        src_oov = src_oov.cuda()

    optimizer.zero_grad()
    # decoder_log_probs的size为(batch_size, trg_seq_len, vocab_size)
    # 返回的probs实际上是log_softmax后的输出
    decoder_log_probs, _, _ = model.forward(src, src_len, trg, src_oov, oov_lists)

    # simply average losses of all the predicitons
    # IMPORTANT, must use logits instead of probs to compute the loss, 
    # otherwise it's super super slow at the beginning (grads of probs are small)!
    start_time = time.time()
    # criterion: NLLLoss(input, target)
    # NLLLoss的使用:
    # input is of size N x C （C为类别数，这里即为vocab size）
    # each element in target has to have 0 <= value < C
    if not opt.copy_attention:
        loss = criterion(decoder_log_probs.contiguous().view(-1, opt.vocab_size), trg_target.contiguous().view(-1))
    else:
        loss = criterion(decoder_log_probs.contiguous().view(-1, opt.vocab_size + max_oov_number), trg_copy_target.contiguous().view(-1))

    print("--loss calculation- %s seconds ---" % (time.time() - start_time))

    # 对loss进行反向传播
    start_time = time.time() 
    loss.backward()
    print("--backward- %s seconds ---" % (time.time() - start_time))

    # 防止梯度过大，对梯度进行裁剪
    if opt.max_grad_norm > 0:
        # 函数用法：torch.nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)
        pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        after_norm = (sum([param.grad.data.norm(2) ** 2 for param in model.parameters() if param.grad is not None])) ** (1.0 / 2)
        if pre_norm != after_norm:
            print('clip grad (%lf -> %lf)' % (pre_norm, after_norm))

    optimizer.step()
    
    # 把loss的值放回cpu（numpy）用于下面的report
    if torch.cuda.is_available():
        loss_value = loss.cpu().data.numpy()
    else:
        loss_value = loss.data.numpy()

    return loss_value, decoder_log_probs

def train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, test_data_loader, opt, logging):
    # SequenceGenerator来自beam_search文件
    seq_generator = SequenceGenerator(model,
                                  eos_id=opt.word2id[vocab.EOS], # EOS = '</s>' EOS_ID = 2
                                  pad_id = opt.word2id[vocab.PAD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length
                                  )

    logging.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available(): # 笔记本电脑Cuda内存太小，只能放在CPU跑，如果要用GPU把not去掉，737行同理
        if isinstance(opt.device_ids, int):
            opt.device_ids = [opt.device_ids]
        logging.info('Running on GPU! devices=%s' % str(opt.device_ids))
        # model = torch.nn.DataParallel(model, device_ids=opt.device_ids)
        model = model.cuda()
    else:
        logging.info('Running on CPU!')

    logging.info('======================  Start Training  =========================')

    checkpoint_names = []
    train_history_losses = []
    valid_history_losses = []
    test_history_losses = []
    # best_loss = sys.float_info.max # for normal training/testing loss (likelihood)
    best_loss = 0.0  # for f-score
    stop_increasing = 0

    train_losses = []
    total_batch = -1
    early_stop_flag = False

    if opt.train_from:
        state_path = opt.train_from.replace('.model', '.state')
        logging.info('Loading training state from: %s' % state_path)
        if os.path.exists(state_path):
            (epoch, total_batch, best_loss, stop_increasing, checkpoint_names, train_history_losses, valid_history_losses,
             test_history_losses) = torch.load(open(state_path, 'rb'))
            opt.start_epoch = epoch

    for epoch in range(opt.start_epoch, opt.epochs + 1): # range左开右闭
        if early_stop_flag:
            break
        
        progbar = Progbar(logger=logging, title='Training', target=len(train_data_loader), batch_size=train_data_loader.batch_size,
                          total_examples=len(train_data_loader.dataset.data))
        
        for batch_idx, batch in enumerate(train_data_loader):
            model.train()
            total_batch += 1
            one2many_batch, one2one_batch = batch
            report_loss = []

            # Training
             # decoder_log_probs的size为(batch_size, trg_seq_len, vocab_size + max_unk_words)
            loss, decoder_log_probs = train(one2one_batch, model, optimizer, criterion, opt)
            # print("decoder log probs size: ", decoder_log_probs.size()) # decoder log probs size:  torch.Size([127, 7, 5044])
            train_losses.append(loss)
            report_loss.append(('train_ml_loss', loss))
            report_loss.append(('PPL', loss))

            # Brief report
            if batch_idx % opt.report_every == 0:
                brief_report(epoch, batch_idx, one2one_batch, loss, decoder_log_probs, opt, logging)
            
            progbar.update(epoch, batch_idx, report_loss)
            
            # Validate and save checkpoint
            if (opt.run_valid_every == -1 and batch_idx == len(train_data_loader) - 1) or\
               (opt.run_valid_every > -1 and total_batch > 1 and total_batch % opt.run_valid_every == 0):
                logging.info('*' * 50)
                logging.info('Run validating and testing @Epoch=%d,#(Total batch)=%d' % (epoch, total_batch))
               
                # 以下函数在evaluate文件中
                valid_score_dict = evaluate_beam_search(seq_generator, valid_data_loader, opt, title='Validating, epoch=%d, batch=%d, total_batch=%d' 
                                                        % (epoch, batch_idx, total_batch), epoch=epoch, predict_save_path=opt.pred_path)
                test_score_dict = evaluate_beam_search(seq_generator, test_data_loader, opt, title='Testing, epoch=%d, batch=%d, total_batch=%d' 
                                                       % (epoch, batch_idx, total_batch), epoch=epoch, predict_save_path=opt.pred_path) 

                checkpoint_names.append('epoch=%d-batch=%d-total_batch=%d' % (epoch, batch_idx, total_batch))

                curve_names = []
                scores = []
                if opt.train_ml:
                    train_history_losses.append(copy.copy(train_losses))
                    scores += [train_history_losses]
                    curve_names += ['Training Loss & Score']
                    train_losses = []
                
                valid_history_losses.append(valid_score_dict)
                test_history_losses.append(test_score_dict)

                scores += [[result_dict[name] for result_dict in valid_history_losses] for name in opt.report_score_names]
                curve_names += ['Valid-' + name for name in opt.report_score_names]
                scores += [[result_dict[name] for result_dict in test_history_losses] for name in opt.report_score_names]
                curve_names += ['Test-' + name for name in opt.report_score_names]

                scores = [np.asarray(s) for s in scores]
                # Plot the learning curve
                plot_learning_curve_and_write_csv(scores=scores,
                                                  curve_names=curve_names,
                                                  checkpoint_names=checkpoint_names,
                                                  title='Training Validation & Test',
                                                  save_path=opt.exp_path + 
                                                  '/[epoch=%d,batch=%d,total_batch=%d]train_valid_test_curve.png' 
                                                  % (epoch, batch_idx, total_batch))

                '''
                determine if early stop training (whether f-score increased, before is if valid error decreased)
                '''
                # report_score_names: default=['f_score@5_exact', 'f_score@5_soft', 'f_score@10_exact', 'f_score@10_soft']
                valid_loss = np.average(valid_history_losses[-1][opt.report_score_names[0]])
                is_best_loss = valid_loss > best_loss
                rate_of_change = float(valid_loss - best_loss) / float(best_loss) if float(best_loss) > 0 else 0.0

                # valid error doesn't increase
                if rate_of_change <= 0:
                    stop_increasing += 1
                else:
                    stop_increasing = 0

                if is_best_loss:
                    logging.info('Validation: update best loss (%.4f --> %.4f), rate of change (ROC)=%.2f' % (
                        best_loss, valid_loss, rate_of_change * 100))
                else:
                    logging.info('Validation: best loss is not updated for %d times (%.4f --> %.4f), rate of change (ROC)=%.2f%%' % (
                        stop_increasing, best_loss, valid_loss, rate_of_change * 100))

                best_loss = max(valid_loss, best_loss)

                # only store the checkpoints that make better validation performances
                if total_batch > 1 and (total_batch % opt.save_model_every == 0 or is_best_loss):  # epoch >= opt.start_checkpoint_at and
                    # Save the checkpoint
                    logging.info('Saving checkpoint to: %s' % os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d.error=%f' 
                                                                    % (opt.exp, epoch, batch_idx, total_batch, valid_loss) + '.model'))
                    torch.save(
                        model.state_dict(),
                        open(os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' 
                                          % (opt.exp, epoch, batch_idx, total_batch) + '.model'), 'wb')
                    )
                    torch.save(
                        (epoch, total_batch, best_loss, stop_increasing, checkpoint_names, 
                         train_history_losses, valid_history_losses, test_history_losses),
                        open(os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' 
                                          % (opt.exp, epoch, batch_idx, total_batch) + '.state'), 'wb')
                    )

                if stop_increasing >= opt.early_stop_tolerance:
                    logging.info('Have not increased for %d validation times, early stop training' % stop_increasing)
                    early_stop_flag = True
                    break
                logging.info('*' * 50)
                
def main():
    opts = init_opts(description='train.py')
    
    logging = init_logging(logger_name='train.py', log_file=opts.log_file, redirect_to_stdout=False)
    logging.info('EXP_PATH : ' + opts.exp_path)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (key, str(value))) for key, value in opts.__dict__.items()]
    
    logging.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        if isinstance(opts.device_ids, int):
            opts.device_ids = [opts.device_ids]
        logging.info('Running on %s! devices=%s' % ('MULTIPLE GPUs' if len(opts.device_ids) > 1 else '1 GPU', str(opts.device_ids)))
    else:
        logging.info('Running on CPU!')
     
    train_data_dataloader, valid_data_dataloader, test_data_dataloader, word2id, id2word, vocab = load_vocab_and_data(opts, logger=logging)
    model = init_model(opts, logging)
    optimizer, criterion = init_optimizer_criterion(model, opts)
    train_model(model, optimizer, criterion, train_data_dataloader, valid_data_dataloader, test_data_dataloader, opts, logging)
    
    
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:56:41 2019

@author: houwenxin
"""

import utils
import torch
import numpy as np
import os
import json
import conf
from report import Progbar
from vocab import UNK

def if_present_keyword(src_str_tokens, keyword_str_tokens):
    match_pos_idx = -1
    for src_start_idx in range(len(src_str_tokens) - len(keyword_str_tokens) + 1):
        match_flag = True
        # 对原文中的每个词作为起点进行遍历，然后对关键词中的每个词进行遍历，如果发现不匹配则设置match为False并跳过这个起点
        for seq_idx, seq_w in enumerate(keyword_str_tokens):
            src_w = src_str_tokens[src_start_idx + seq_idx]
            if src_w != seq_w:
                match_flag = False
                break
        if match_flag:
            match_pos_idx = src_start_idx
            break
    return match_flag, match_pos_idx
def if_present_duplicate_keywords(src_str, trgs_str, check_duplicate=True):
    src_to_match = src_str

    present_indices = []
    present_flags = []
    phrase_set = set()  
    # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for trg_str in trgs_str:
        trg_to_match = trg_str

        # check if the phrase appears in source text
        # iterate each word in source
        match_flag, match_pos_idx = if_present_keyword(src_to_match, trg_to_match)

        # check if it is duplicate, if true then ignore it
        if check_duplicate and "_".join(trg_to_match) in phrase_set:
            present_flags.append(False)
            present_indices.append(match_pos_idx)
            continue
        else:
            # if it reaches the end of source and no match, means it doesn't appear in the source
            present_flags.append(match_flag)
            present_indices.append(match_pos_idx)

        phrase_set.add("_".join(trg_to_match))

    assert len(present_flags) == len(present_indices)
    return present_flags, present_indices

def process_predseqs(pred_seqs, oov, id2word, opt):
    '''
    :param pred_seqs:
    :param src_str:
    :param oov:
    :param id2word:
    :param opt:
    :return:
    '''
    processed_seqs = []
    if_valid = []

    for seq in pred_seqs:
        # print('-' * 50)
        # print('seq.sentence: ' + str(seq.sentence))
        # print('oov: ' + str(oov))
        #
        #for x in seq.sentence[:-1]:
        #    if x >= opt.vocab_size and len(oov)==0:
        #        print('ERROR')

        # convert to words and remove the EOS token
        seq_sentence_np = [int(x) for x in seq.sentence]

        processed_seq = [id2word[x] if x < opt.vocab_size else oov[x - opt.vocab_size] for x in seq_sentence_np[:-1]]
        # print('processed_seq: ' + str(processed_seq))

        # print('%s - %s' % (str(seq.sentence[:-1]), str(processed_seq)))

        keep_flag = True

        if len(processed_seq) == 0:
            keep_flag = False

        if keep_flag and any([w == UNK for w in processed_seq]):
            keep_flag = False
        """
        if keep_flag and any([w == '.' or w == ',' for w in processed_seq]):
            keep_flag = False
        """
        if_valid.append(keep_flag)
        processed_seqs.append((seq, processed_seq, seq.score))

    unzipped = list(zip(*(processed_seqs)))
    processed_seqs, processed_str_seqs, processed_scores = unzipped if len(processed_seqs) > 0 and len(unzipped) == 3 else ([], [], [])

    assert len(processed_seqs) == len(processed_str_seqs) == len(processed_scores) == len(if_valid)
    return if_valid, processed_seqs, processed_str_seqs, processed_scores

def get_match_result(true_seqs, pred_seqs, type='exact'):
    '''
    :param type: 'exact' or 'partial'
    :‘exact’ means predictoin and ground-truth are exactly the same
    :'partial' evaluate their similarity using Jaccard coefficient
    :return: match_score_list
    '''

    match_score_list = np.asarray([0.0] * len(pred_seqs), dtype='float32')

    for pred_idx, pred_seq in enumerate(pred_seqs):
        if type == 'exact': # exact表示精确匹配，要求预测和真实的关键词一模一样
            match_score_list[pred_idx] = 0
            for true_idx, true_seq in enumerate(true_seqs):
                match = True
                if len(pred_seq) != len(true_seq): # 如果长度都不相等就直接跳过
                    continue
                for pred_w, true_w in zip(pred_seq, true_seq):
                    # if one two words are not same, match fails
                    if pred_w != true_w:
                        match = False
                        break
                # if every word in pred_seq matches one true_seq exactly, match succeeds
                if match:
                    match_score_list[pred_idx] = 1
                    break
        elif type == 'partial':
            max_similarity = 0.
            pred_seq_set = set(pred_seq)
            # 使用Jaccard相似指数作为模糊匹配的衡量指标
            # Jaccard相似指数用来度量两个集合之间的相似性，它被定义为两个集合交集的元素个数除以并集的元素个数
            for true_idx, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                jaccard = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                if jaccard > max_similarity:
                    max_similarity = jaccard
            match_score_list[pred_idx] = max_similarity
        """ bleu暂时不考虑
        elif type == 'bleu':
            # account for the match of subsequences, like n-gram-based (BLEU) or LCS-based
            match_score[pred_id] = bleu(pred_seq, true_seqs, [0.1, 0.3, 0.6])
        """
    return match_score_list

def post_process_predseqs(seqs, num_oneword_seq=1):
    processed_seqs = []

    # -1 means no filter applied
    if num_oneword_seq == -1:
        return seqs

    for seq, str_seq, score in zip(*seqs):
        keep_flag = True

        if len(str_seq) == 1 and num_oneword_seq <= 0:
            keep_flag = False

        if keep_flag:
            processed_seqs.append((seq, str_seq, score))
            # update the number of one-word sequeces to keep
            if len(str_seq) == 1:
                num_oneword_seq -= 1

    unzipped = list(zip(*(processed_seqs)))
    if len(unzipped) != 3:
        return ([], [], [])
    else:
        return unzipped

def evaluate(match_score_list, predicted_list, true_list, topk=5):
    '''
    :topk: if number of predicted keywords exceeds k, then truncate it to k 
    '''
    if len(match_score_list) > topk:
        match_score_list = match_score_list[:topk]
    if len(predicted_list) > topk:
        predicted_list = predicted_list[:topk]

    # micropk: 查准率 # micrork: 查全率
    micropk = float(sum(match_score_list)) / float(len(predicted_list)) if len(predicted_list) > 0 else 0.0
    micrork = float(sum(match_score_list)) / float(len(true_list)) if len(true_list) > 0 else 0.0
    # 计算micro F1分数
    if micropk + micrork > 0:
        microf1 = float(2 * (micropk * micrork)) / (micropk + micrork)
    else:
        microf1 = 0.0

    return micropk, micrork, microf1

def evaluate_beam_search(sequence_generator, data_loader, opt, title='', epoch=1, predict_save_path=None):
    logger = conf.init_logging(title, predict_save_path + '/%s.log' % title, redirect_to_stdout=False)
    progbar = Progbar(logger=logger, title=title, target=len(data_loader.dataset.data), batch_size=data_loader.batch_size,
                      total_examples=len(data_loader.dataset.data))

    topk_range = [5, 10]
    score_names = ['precision', 'recall', 'f_score']

    example_idx = 0
    score_dict = {}  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    for i, batch in enumerate(data_loader):
        
        one2many_batch, one2one_batch = batch
        """
        one2many:
            text_lists_one2many, text_one2many_lens, keywords_lists_one2many, _, 
                    keywords_oov_target_lists_one2many, text_oov_lists_one2many, oov_lists_one2many, 
                    text_tokens_lists, keywords_tokens_lists
        """
        src_lists, src_lens, trg_lists, _, trg_copy_target_lists, src_oov_lists, oov_lists, src_str_lists, trg_str_lists = one2many_batch

        if torch.cuda.is_available():
            src_lists = src_lists.cuda()
            src_oov_lists = src_oov_lists.cuda()

        print("batch size - %s" % str(src_lists.size(0)))
        print("src size - %s" % str(src_lists.size()))
        print("target size - %s" % len(trg_copy_target_lists))

        pred_seq_list = sequence_generator.beam_search(src_lists, src_lens, src_oov_lists, oov_lists, opt.word2id)

        '''
        process each example in current batch
        '''
        for src, src_str, trg, trg_str_seqs, trg_copy, pred_seq, oov in zip(src_lists, 
                                                                            src_str_lists, 
                                                                            trg_lists, 
                                                                            trg_str_lists, 
                                                                            trg_copy_target_lists, 
                                                                            pred_seq_list, 
                                                                            oov_lists):
            logger.info('======================  Batch: %d =========================' % (i))
            print_out = ''
            print_out += '[Source][%d]: %s \n' % (len(src_str), ' '.join(src_str))
            src = src.cpu().data.numpy() if torch.cuda.is_available() else src.data.numpy()
            print_out += '\nSource Input: \n %s\n' % (' '.join([opt.id2word[x] for x in src[:len(src_str) + 5]]))
            print_out += 'Real Target String [%d] \n\t\t%s \n' % (len(trg_str_seqs), trg_str_seqs)
            print_out += 'Real Target Input:  \n\t\t%s \n' % str([[opt.id2word[x] for x in t] for t in trg])
            print_out += 'Real Target Copy:   \n\t\t%s \n' % str([[opt.id2word[x] if x < opt.vocab_size else oov[x - opt.vocab_size] for x in t] for t in trg_copy])
            trg_str_is_present_flags, _ = if_present_duplicate_keywords(src_str, trg_str_seqs)
            
            # ignore the cases that there's no present phrases
            if opt.must_appear_in_src and np.sum(trg_str_is_present_flags) == 0:
                logger.error('found no present targets')
                continue
            
            print_out += '[GROUND-TRUTH] #(present)/#(all targets)=%d/%d\n' % (sum(trg_str_is_present_flags), len(trg_str_is_present_flags))
            print_out += '\n'.join(['\t\t[%s]' % ' '.join(phrase) if is_present else '\t\t%s' % ' '.join(phrase) for phrase, is_present in zip(trg_str_seqs, trg_str_is_present_flags)])
            print_out += '\noov_list:   \n\t\t%s \n' % str(oov)

            # 1st filtering
            pred_is_valid_flags, processed_pred_seqs, processed_pred_str_seqs, processed_pred_score = process_predseqs(pred_seq, oov, opt.id2word, opt)
            # 2nd filtering: if filter out phrases that don't appear in text, and keep unique ones after stemming
            if opt.must_appear_in_src:
                pred_is_present_flags, _ = if_present_duplicate_keywords(src_str, processed_pred_str_seqs)
                filtered_trg_str_seqs = np.asarray(trg_str_seqs)[trg_str_is_present_flags]
            else:
                pred_is_present_flags = [True] * len(processed_pred_str_seqs)

            valid_and_present = np.asarray(pred_is_valid_flags) * np.asarray(pred_is_present_flags)
            match_score_list = get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=processed_pred_str_seqs)
            print_out += '[PREDICTION] #(valid)=%d, #(present)=%d, #(valid&present)=%d, #(all)=%d\n' \
                                % (sum(pred_is_valid_flags), sum(pred_is_present_flags), sum(valid_and_present), len(pred_seq))
            print_out += ''
            '''
            Print and export predictions
            '''
            preds_out = ''
            for p_id, (seq, word, score, match, is_valid, is_present) in enumerate(
                    zip(processed_pred_seqs, processed_pred_str_seqs, processed_pred_score, match_score_list, 
                                                            pred_is_valid_flags, pred_is_present_flags)):
                preds_out += '%s\n' % (' '.join(word))
                if is_present:
                    print_phrase = '[%s]' % ' '.join(word)
                else:
                    print_phrase = ' '.join(word)

                if is_valid:
                    print_phrase = '*%s' % print_phrase

                if match == 1.0:
                    correct_str = '[correct!]'
                else:
                    correct_str = ''
                if any([t >= opt.vocab_size for t in seq.sentence]):
                    copy_str = '[copied!]'
                else:
                    copy_str = ''

                print_out += '\t\t[%.4f]\t%s \t %s %s%s\n' % (-score, print_phrase, str(seq.sentence), correct_str, copy_str)

            '''
            Evaluate predictions with regard to different filterings and metrics
            '''
            processed_pred_seqs = np.asarray(processed_pred_seqs)[valid_and_present]
            filtered_processed_pred_str_seqs = np.asarray(processed_pred_str_seqs)[valid_and_present]
            filtered_processed_pred_score = np.asarray(processed_pred_score)[valid_and_present]

            # 3rd round filtering (one-word phrases)
            num_oneword_seq = -1
            filtered_pred_seq, filtered_pred_str_seqs, filtered_pred_score = post_process_predseqs((processed_pred_seqs, 
                                                                                                    filtered_processed_pred_str_seqs, 
                                                                                                    filtered_processed_pred_score), 
                                                                                                    num_oneword_seq)

            match_score_list_exact = get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=filtered_pred_str_seqs, type='exact')
            match_score_list_soft = get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=filtered_pred_str_seqs, type='partial')

            assert len(filtered_pred_seq) == len(filtered_pred_str_seqs) == len(filtered_pred_score) == \
                            len(match_score_list_exact) == len(match_score_list_soft)

            print_out += "\n =======================Keywords in Text================================="
            print_pred_str_seqs = [" ".join(item) for item in filtered_pred_str_seqs]
            print_trg_str_seqs = [" ".join(item) for item in filtered_trg_str_seqs]
            print_out += "\n PREDICTION: " + " / ".join(print_pred_str_seqs)
            print_out += "\n GROUND TRUTH: " + " / ".join(print_trg_str_seqs)

            for topk in topk_range:
                results_exact = evaluate(match_score_list_exact, filtered_pred_str_seqs, filtered_trg_str_seqs, topk=topk)
                for k, v in zip(score_names, results_exact):
                    if '%s@%d_exact' % (k, topk) not in score_dict:
                        score_dict['%s@%d_exact' % (k, topk)] = []
                    score_dict['%s@%d_exact' % (k, topk)].append(v)
                
                print_out += "\n ------------------------------------------------- EXACT, k=%d" % (topk)
                print_out += "\n --- batch precision, recall, fscore: " + str(results_exact[0]) + " , " + str(results_exact[1]) + " , " + str(results_exact[2])
                print_out += "\n --- total precision, recall, fscore: " + str(np.average(score_dict['precision@%d_exact' % (topk)])) + " , " +\
                            str(np.average(score_dict['recall@%d_exact' % (topk)])) + " , " +\
                            str(np.average(score_dict['f_score@%d_exact' % (topk)]))

            for topk in topk_range:
                results_soft = evaluate(match_score_list_soft, filtered_pred_str_seqs, filtered_trg_str_seqs, topk=topk)
                for k, v in zip(score_names, results_soft):
                    if '%s@%d_soft' % (k, topk) not in score_dict:
                        score_dict['%s@%d_soft' % (k, topk)] = []
                    score_dict['%s@%d_soft' % (k, topk)].append(v)
                
                print_out += "\n ------------------------------------------------- SOFT, k=%d" % (topk)
                print_out += "\n --- batch precision, recall, fscore: " + \
                            str(results_soft[0]) + " , " + str(results_soft[1]) + " , " + str(results_soft[2])
                print_out += "\n --- total precision, recall, fscore: " + \
                            str(np.average(score_dict['precision@%d_soft' % (topk)])) + " , " +\
                            str(np.average(score_dict['recall@%d_soft' % (topk)])) + " , " +\
                            str(np.average(score_dict['f_score@%d_soft' % (topk)]))

            print_out += "\n ======================================================="
            logger.info(print_out)

            '''
            write predictions to disk
            '''
            if predict_save_path:
                if not os.path.exists(os.path.join(predict_save_path, title + '_detail')):
                    os.makedirs(os.path.join(predict_save_path, title + '_detail'))
                with open(os.path.join(predict_save_path, title + '_detail', str(example_idx) + '_print.txt'), 'w') as f_:
                    f_.write(print_out)
                with open(os.path.join(predict_save_path, title + '_detail', str(example_idx) + '_prediction.txt'), 'w') as f_:
                    f_.write(preds_out)

                out_dict = {}
                out_dict['src_str'] = src_str
                out_dict['trg_str'] = trg_str_seqs
                out_dict['trg_present_flag'] = trg_str_is_present_flags
                out_dict['pred_str'] = processed_pred_str_seqs
                out_dict['pred_score'] = [float(s) for s in processed_pred_score]
                out_dict['present_flag'] = pred_is_present_flags
                out_dict['valid_flag'] = pred_is_valid_flags
                out_dict['match_flag'] = [float(m) for m in match_score_list]

                for k,v in out_dict.items():
                    out_dict[k] = list(v)
                    # print('len(%s) = %d' % (k, len(v)))

                # print(out_dict)

                assert len(out_dict['trg_str']) == len(out_dict['trg_present_flag'])
                assert len(out_dict['pred_str']) == len(out_dict['present_flag']) \
                       == len(out_dict['valid_flag']) == len(out_dict['match_flag']) == len(out_dict['pred_score'])

                with open(os.path.join(predict_save_path, title + '_detail', str(example_idx) + '.json'), 'w') as f_:
                    f_.write(json.dumps(out_dict))

            progbar.update(epoch, example_idx, [('f_score@5_exact', np.average(score_dict['f_score@5_exact'])),
                                                ('f_score@5_soft', np.average(score_dict['f_score@5_soft'])),
                                                ('f_score@10_exact', np.average(score_dict['f_score@10_exact'])),
                                                ('f_score@10_soft', np.average(score_dict['f_score@10_soft'])),])

            example_idx += 1

    if predict_save_path:
        # export scores. Each row is scores (precision, recall and f-score) of different way of filtering predictions (how many one-word predictions to keep)
        with open(predict_save_path + os.path.sep + title + '_result.csv', 'w') as result_csv:
            csv_lines = []
            for mode in ["exact", "soft"]:
                for topk in topk_range:
                    csv_line = ""
                    for k in score_names:
                        csv_line += ',%f' % np.average(score_dict['%s@%d_%s' % (k, topk, mode)])
                    csv_lines.append(csv_line + '\n')

            result_csv.writelines(csv_lines)

    # precision, recall, f_score = macro_averaged_score(precisionlist=score_dict['precision'], recalllist=score_dict['recall'])
    # logging.info("Macro@5\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f " % (np.average(score_dict['precision@5']), np.average(score_dict['recall@5']), np.average(score_dict['f1score@5'])))
    # logging.info("Macro@10\n\t\tprecision %.4f\n\t\tmacro recall %.4f\n\t\tmacro fscore %.4f " % (np.average(score_dict['precision@10']), np.average(score_dict['recall@10']), np.average(score_dict['f1score@10'])))
    # precision, recall, f_score = evaluate(true_seqs=target_all, pred_seqs=prediction_all, topn=5)
    # logging.info("micro precision %.4f , micro recall %.4f, micro fscore %.4f " % (precision, recall, f_score))

    for key, value in score_dict.items():
        print('#(%s) = %d' % (key, len(value)))

    return score_dict
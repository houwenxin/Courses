# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:52:51 2019

@author: houwenxin
"""
import torch
import os
import utils
from vocab import EOS
from dataloader import KeywordDataset, KeywordDataLoader
from model import init_model
import conf
from beam_search import SequenceGenerator
from evaluate import evaluate_beam_search

def load_vocab_and_testsets(opt):
    print("Loading vocab from disk: %s" % (opt.vocab))
    word2id, id2word, vocab = torch.load(opt.vocab, 'rb')
    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab
    print('#(vocab)=%d' % len(vocab))
    print('#(vocab used)=%d' % opt.vocab_size)

    pin_memory = torch.cuda.is_available()
    test_one2many_loaders = []

    for testset_name in opt.test_dataset_names:
        print("Loading test dataset %s" % testset_name)
        testset_path = os.path.join(opt.test_dataset_root_path, testset_name + '.test.one2many.pth')
        test_one2many = torch.load(testset_path, 'wb')
        test_one2many_dataset = KeywordDataset(test_one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)
        test_one2many_loader = KeywordDataLoader(dataset=test_one2many_dataset,
                                                   collate_fn=test_one2many_dataset.collate_fn_one2many,
                                                   num_workers=opt.batch_workers,
                                                   max_text_num=opt.beam_search_batch_example,
                                                   max_batch_size=opt.beam_search_batch_size,
                                                   pin_memory=pin_memory,
                                                   shuffle=False)

        test_one2many_loaders.append(test_one2many_loader)
        print('#(test data size:  #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d' % (len(test_one2many_loader.dataset), test_one2many_loader.one2one_number(), len(test_one2many_loader)))
        print('*' * 50)

    return test_one2many_loaders, word2id, id2word, vocab


def main():
    opt = conf.init_opts(description='predict.py')
    logger = conf.init_logging('predict', os.path.join(opt.exp_path, 'predict_output.log'), redirect_to_stdout=False)

    print('EXP_PATH : ' + opt.exp_path)

    print('Parameters:')
    [print('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    print('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        if isinstance(opt.device_ids, int):
            opt.device_ids = [opt.device_ids]
        print('Running on %s! devices=%s' % ('MULTIPLE GPUs' if len(opt.device_ids) > 1 else '1 GPU', str(opt.device_ids)))
    else:
        print('Running on CPU!')

    try:
        test_data_loaders, word2id, id2word, vocab = load_vocab_and_testsets(opt)
        model = init_model(opt, logger)
        generator = SequenceGenerator(model,
                                      eos_id=opt.word2id[EOS],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length
                                      )

        for testset_name, test_data_loader in zip(opt.test_dataset_names, test_data_loaders):
            print('Evaluating %s' % testset_name)
            evaluate_beam_search(generator, test_data_loader, opt,
                                 title='test_%s' % testset_name,
                                 predict_save_path=opt.pred_path
                                 )

    except Exception as e:
        logger.error(e, exc_info=True)

if __name__ == '__main__':
    main()

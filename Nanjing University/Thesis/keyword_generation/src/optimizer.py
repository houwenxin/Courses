# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:16:47 2019

@author: houwenxin
"""

import torch
import vocab
from torch.optim import Adam

def init_optimizer_criterion(model, opts):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """
    '''
    if not opt.copy_attention:
        weight_mask = torch.ones(opt.vocab_size).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size)
    else:
        weight_mask = torch.ones(opt.vocab_size + opt.max_unk_words).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size + opt.max_unk_words)
    weight_mask[opt.word2id[pykp.IO.PAD_WORD]] = 0
    criterion = torch.nn.NLLLoss(weight=weight_mask)

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
    '''

    criterion = torch.nn.NLLLoss(ignore_index=opts.word2id[vocab.PAD])

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opts.learning_rate)

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    return optimizer, criterion
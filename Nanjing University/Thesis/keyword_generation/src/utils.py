# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:27:12 2019

@author: houwenxin
"""

def tally_parameters(model, logging):
    # if logging.getLogger() == None:
    #    printer = print
    # else:
    #    printer = logging.getLogger().info
    printer = logging.info
    n_params = sum([p.nelement() for p in model.parameters()])
    printer('Model name: %s' % type(model).__name__)
    printer('number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    printer('encoder: %d' % enc)
    printer('decoder: %d' % dec)
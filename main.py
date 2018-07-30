#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:36:05 2018

@author: exist
"""

from chainer import optimizers

from network import VegeCNN

def load_data():
    pass

def train(model_class):
    model = VegeCNN()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
#    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    
    
    
if __name__ == '__main__':
    load
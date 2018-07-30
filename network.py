#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:36:05 2018

@author: exist
"""

import chainer.functions as F
import chainer.links as L
from chainer import Chain

class VegeCNN(Chain):
    def __init__(self, label_count, train=True):
        super(VegeCNN, self).__init__(
                conv1 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True),
                conv2 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True),
                l1=L.Linear(None, label_count, nobias=True)
                )
        self.train = train
    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)))
        h1 = F.max_pooling_2d(F.relu(self.conv2(h)))
        h2 = self.l1(h1)
        return h2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:36:05 2018

@author: exist
"""
import os
import time

import pandas as pd
import numpy as np
from PIL import Image

from chainer import optimizers, iterators, Variable, serializers
from chainer import functions as F
from network import VegeCNN

PATH_TO_TRAIN_IMAGE = os.path.join('datas', 'processed', 'processed_train_images')
PATH_TO_TRAIN_DATA = os.path.join('datas', 'given', 'train_master.tsv')
PATH_TO_MODEL = os.path.join('models', 'VegeCNN')
label_count = 0

def load_data(path_to_train_images, path_to_train_data):
    print('loading train data ...')
    df = pd.read_table(path_to_train_data, index_col=0)
    X = []
    y = []
    

    for row in df.iterrows():
        file_name, cate_id = row[0], row[1]
        try:
            im = Image.open(os.path.join(path_to_train_images, file_name))
            assert im.size == (256, 256), "Image does not preprocessed ...? \n Please check image:{0}".format(f)
            X.append(np.array(im))
            y.append(cate_id)
        except Exception as e:
            print(str(e))
            
            X = np.asarray(X, dtype=np.float32)
            y = np.array(y)
            print('done.')
            return X, y
        
    X = np.array(X)
    y = np.array(y)
    print('done.')
    return X, y

def train_model(X_train, y_train):
    print('training...')
    loss_sum = 0
    acc_sum = 0
    train_count = X_train.shape[0]
    model = VegeCNN(55)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
#    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    
    
    train_iter_X = iterators.SerialIterator(X_train, batch_num)
    train_iter_y = iterators.SerialIterator(y_train, batch_num)
    
    start = time.time()
#    while train_iter.epoch < EPOCH:
    batch_X = np.array(train_iter_X.next())
    batch_y = np.array(train_iter_y.next())
    X = Variable(batch_X)
    t = Variable(batch_y)
    y = model(X)
    
    loss = F.soft_max_entropy(y, t)
    acc  = F.accuracy(y, t)
    model.cleargrads()
    loss.backward()
    optimizer.update()
    
    loss_sum += loss
    acc_sum += acc
#    if train_iter.is_new_epoch:net
#    print('epoch: ', 1)
    print('train mean loss: {:.2f}, accuracy: {:.2f}'.format( loss_sum / train_count, acc_sum / train_count))
             

    end = time.time()
    elapsd_time = end-start
    print("elapsd_time:{0}".format(elapsd_time))
    return model
    
    
#学習パラメータの保存
def save_model(model, name):
    print('saving the model ...')
    serializers.save_hdf5(name+'.model', model)
    print('done.')

    
if __name__ == '__main__':
    sep_num = 3000
    batch_num = 512
    EPOCH = 100
    model_name = 'VegeCNN'
    
    X, y = load_data(PATH_TO_TRAIN_IMAGE, PATH_TO_TRAIN_DATA)
    N = X.shape[0]
    index = np.random.permutation(N)
    index_train, index_test = index[:sep_num], index[sep_num:]
    
    X_train, X_test = X[index_train], X[index_test]
    y_train, y_test = y[index_train], y[index_test]
    model = train_model(X_train, y_train)
    
#    save_model(model, PATH_TO_MODEL)
    
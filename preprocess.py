#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:38:28 2018

@author: exist
"""


import os
import zipfile
from PIL import Image

ZIP_FILE_NAME_TRAIN = 'train_1'
ZIP_FILE_NAME_TEST = 'test'

PATH_TO_GIVEN_DATA = os.path.join('datas','given')
PATH_TO_MY_DATA = os.path.join('datas','processed')
if not os.path.exists(PATH_TO_MY_DATA):
    os.mkdir(PATH_TO_MY_DATA)

PATH_TO_TRAIN_IMAGES = os.path.join(PATH_TO_MY_DATA, ZIP_FILE_NAME_TRAIN)
PATH_TO_TEST_IMAGES = os.path.join(PATH_TO_MY_DATA, 'test_images')

PATH_TO_DST_TRAIN_IMAGES = os.path.join(PATH_TO_MY_DATA, 'processed_train_images')
PATH_TO_DST_TEST_IMAGES = os.path.join(PATH_TO_MY_DATA, 'processed_test_images')
if not os.path.exists(PATH_TO_DST_TRAIN_IMAGES):
    os.mkdir(PATH_TO_DST_TRAIN_IMAGES)
if not os.path.exists(PATH_TO_DST_TEST_IMAGES):
    os.mkdir(PATH_TO_DST_TEST_IMAGES)

def extract_zipfile(src, dst):
    print('extracting '+src+' to '+dst+' ...')
    with zipfile.ZipFile(src, 'r') as zip_file:
        zip_file.extractall(dst)
    print('done.')

def image_verification(path_to_images):
    files = os.listdir(path_to_images)
    for f in files:
        image_path = os.path.join(path_to_images, f)
        f, h, w, mode = get_image_meta(image_path)
        assert h > 256 and w > 256 ,"The image info is invaild.file:{0}({1},{2})".format(image_path, h, w)
        
    print('Image verification is passed.(*^^)b')

def get_image_meta(path_to_image):
    im = Image.open(path_to_image)
    f = im.format
    h, w = im.size
    mode = im.mode
    
    return f, h, w, mode

def preprocess_images(path_to_images, path_to_dst_images):
    files = os.listdir(path_to_images)
    for f in files:
        im = Image.open(os.path.join(path_to_images, f))
        im = im.resize((256, 256), Image.LANCZOS)
        im.save(os.path.join(path_to_dst_images, f))



if __name__ == '__main__':
    
    if not os.path.exists(PATH_TO_TRAIN_IMAGES):
        extract_zipfile(os.path.join(PATH_TO_GIVEN_DATA, ZIP_FILE_NAME_TRAIN+'.zip'), PATH_TO_MY_DATA)
#    if not os.path.exists(PATH_TO_TEST_IMAGES):
#        extract_zipfile(os.path.join(PATH_TO_GIVEN_DATA, ZIP_FILE_NAME_TEST'.zip'), PATH_TO_MY_DATA)
    image_verification(PATH_TO_TRAIN_IMAGES)
#    image_verification(PATH_TO_TEST_IMAGES)
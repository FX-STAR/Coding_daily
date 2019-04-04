#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function

import numpy as numpy
import cv2,os,sys

caffe_root = '../../'
os.chdir(caffe_root)
sys.path.insert(0,'python')
import caffe
caffe.set_mode_gpu()

import os
print(os.getcwd())
model_def = './data/Occ/regress_deploy.prototxt'
model_weights = './data/Occ/regress_112_result/occ_iter_21000.caffemodel'


net = caffe.Net(model_def, model_weights, caffe.TEST)

IMAGE_SIZE = (112, 112)


net.blobs['data'].reshape(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))       # (h,w,c)--->(c,h,w)
transformer.set_mean('data', numpy.array([104,117,123])) # BGR
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # RGB--->BGR

image_list = sys.argv[1]

import time 



MAE = 0
NUM = 0
TIME = 0
with open(image_list, 'r') as f:
    for line in f.readlines():
        NUM += 1
        filename = os.path.join('./data/Occ/data', line.split()[0])
        im = cv2.imread(filename)
        if im is None:
            continue
        image = caffe.io.load_image(filename)
        tic = time.time()
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        pred = output['pred'].reshape(1)[0]
        gt = float(line.split()[1])
        mae = abs(pred-gt)
        MAE += mae
        TIME += time.time()-tic
        print('gt:{} mae:{} time:{}'.format(gt, mae, time.time()-tic))
        print('The predicted score for {} is {}'.format(filename, output['pred'].reshape(1)[0]))

print('AVG MAE:{}    AVG TIME:{}'.format(MAE/NUM, TIME/NUM))
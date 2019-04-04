#!/usr/bin/env python
# coding=utf-8
import os
from PIL import Image
import sys
import numpy as np
import h5py

IMAGE_SIZE = (112, 112)

filename = sys.argv[1]
setname, ext = filename.split('.')

with open(filename, 'r') as f:
    lines = f.readlines()

# anno = list()
# anno = sorted(lines, key= lambda x: float(x.split()[1]), reverse=True)
# for i in anno:
#     print(i)


np.random.shuffle(lines)

label_path = 'label_{}x{}'.format(IMAGE_SIZE[0], IMAGE_SIZE[1])
if not os.path.exists(label_path):
    os.makedirs(label_path)

for seg in range(2):#int(len(lines)/1000)):
    if seg==0:
       lines_seg =lines[:int(len(lines)/2)]#[1000*seg:1000*seg+1000-1]
    else:
       lines_seg = lines[int(len(lines)/2):]
    sample_size = len(lines_seg)
    imgs = np.zeros((sample_size, 3,)+ IMAGE_SIZE, dtype=np.float32)
    scores = np.zeros(sample_size, dtype=np.float32)
    h5_filename = '{}/{}_{}.h5'.format(label_path, setname, seg)
    with h5py.File(h5_filename, 'a') as h:
        for i, line in enumerate(lines_seg):
            image_name, score = line[:-1].split()
            img = Image.open(os.path.join('data',image_name))
            img = img.resize(IMAGE_SIZE, Image.BILINEAR)
            img = np.array(img, dtype=np.float32) # img:(h,w,3) RGB
            img = img[:, :, ::-1] # img: (h,w,3) BGR
            img -= np.array((104, 117, 123)) #  BGR
            img = img.transpose((2, 0, 1)) # img:(3,h,w)
            img = img.reshape((1, )+img.shape)

            imgs[i] = img
            scores[i] = float(score)
            if (i+1) % 100 == 0:
                print('processed {} images!'.format(i+1))
        h.create_dataset('data', data=imgs)
        h.create_dataset('score', data=scores)


    with open('{}/{}_h5.txt'.format(label_path, setname), 'a+') as f:
        f.write(h5_filename)
        f.write('\n')



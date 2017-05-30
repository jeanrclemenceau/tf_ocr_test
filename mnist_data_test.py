# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:41:23 2017

@author: clemenj
"""

from mnist import MNIST

mndata = MNIST.load('train-images.idx3-ubyte','train-labels.idx1-ubyte')
mn_test_data = MNIST.load('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')


MNIST.display(mndata[0][0])

for i in range(5):
    increment=10
    idx = i*increment
    img = mndata[0][i:i+increment]
    lab = mndata[1][i:i+increment]
    print(i + len(img) + len(lab))

for i in range(5):
    print(mndata[1][i:i+5])

for i in range(len(img)):
    MNIST.display(img[i])

MNIST.display(img[1])
MNIST.display(img[2])
MNIST.display(img[3])

def next_batch(batch_size,cur_idx,data):
    if (cur_idx+batch_size < len(data[0]) ): #return batch size
        end_idx = cur_idx+batch_size
        return  [data[0][cur_idx:end_idx],data[1][cur_idx:end_idx].tolist(),end_idx]
    elif (cur_idx < len(data[0])): #return remainder
        return  [data[0][cur_idx:],data[1][cur_idx:].tolist(),len(data[0])]
    else:
        return [None,None,cur_idx]

idx=0
for i in range(100):
    batch_img,batch_lbl,idx = next_batch(10,idx,mndata)

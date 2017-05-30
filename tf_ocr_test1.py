# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:41:23 2017

@author: clemenj
"""
import tensorflow as tf
from mnist import MNIST

#define batching function for test
def next_batch(batch_size,cur_idx,data):
    if (cur_idx+batch_size < len(data[0]) ): #return batch size
        end_idx = cur_idx+batch_size
        return  [data[0][cur_idx:end_idx],data[1][cur_idx:end_idx].tolist(),end_idx]
    elif (cur_idx < len(data[0])): #return remainder
        return  [data[0][cur_idx:],data[1][cur_idx:].tolist(),len(data[0])]
    else:
        return [None,None,cur_idx]

#initialize variables and placeholders
X = tf.placeholder(tf.float32, [None,28,28,1]) #image data: [Batch_size, X-pixels,Y-pixels,#of colors(gray/rgb)]
W= tf.Variable(tf.zeros([784,10])) #Weights [# of pixels, # of neurons(possible values)]
b= tf.Variable(tf.zeros([10])) #biases [additional weight added per possible image value]
init = tf.global_variables_initializer();

#model: X*W + b (broadcasted  additon). Returns probablities for each neuron
Y=tf.nn.softmax(tf.matmul(tf.reshape(X,[-1,784]),W)+b)

#placeholder for correct answers
Y_=tf.placeholder(tf.float32,[None,10])

#loss function: minimizes distance between truth and test
cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))

#% of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

#define how much distance is minimized per training iteration
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

#start session (instantiate network)
sess = tf.Session()
sess.run(init)

#load training and test set
mn_full_data = MNIST.load('train-images.idx3-ubyte','train-labels.idx1-ubyte')
mn_train_data = MNIST.load('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')

idx=0
for i in range(100):
    #load a batch of images and correct answers
    batch_X,batch_Y,idx = next_batch(100,idx,mn_train_data)
    train_data={X: batch_X, Y_:batch_Y}

    #train
    sess.run(train_step,feed_dict=train_data)

#determine training success
a,c= sess.run([accuracy,cross_entropy],feed=train_data)

#Run full test
#test_data = {X:mn_full_data[0],Y_:mn_full_data[1]}
#a,c= sess.run([accuracy,cross_entropy],feed=test_data)



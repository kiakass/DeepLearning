# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:34:59 2019

@author: Administrator
"""


import tensorflow as tf

from matplotlib import pyplot as plt
import numpy as np

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


x_data = [[[ 1.1, 0.4, 0.7],[ 1.4, 0.3, 1.6]],
          [[ 1.4, 0.3, 1.6],[ 1.6, 0.7, 0.2]],
          [[ 1.6, 0.7, 0.2],[ 1.7, 0.5, 0.9]],
          [[ 1.7, 0.5, 0.9],[ 1.2, 0.4, 0.2]],
          [[ 1.2, 0.4, 0.2],[ 1.3, 0.5, 0.3]],
          [[ 1.3, 0.5, 0.3],[ 1.3, 0.1, 1.0]],
          [[ 1.3, 0.1, 1.0],[ 1.6, 0.1, 0.1]],
          [[ 1.6, 0.1, 0.1],[ 1.8, 0.2, 1.3]],
          [[ 1.8, 0.2, 1.3],[ 1.5, 0.6, 0.6]],
          [[ 1.5, 0.6, 0.6],[ 1.4, 0.5, 0.5]]]

y_data = [[[ 0.7, 1.9]],
          [[ 1.6, 2.0]],
          [[ 0.2, 2.2]],
          [[ 0.9, 2.3]],
          [[ 0.2, 2.3]],
          [[ 0.3, 2.2]],
          [[ 1.0, 2.1]],
          [[ 0.1, 2.5]],
          [[ 1.3, 2.5]],
          [[ 0.6, 3.0]]]
'''
W = [[[x,x]
      [x,x]
      [x,x]]
     [[x,x,x]
      [x,x,x]
      [x,x,x]]]

Y = [[[y],[y]],
     [[y],[y]]
]
'''
tf.shape(x_data).eval()
tf.shape(y_data).eval()

X = tf.placeholder(tf.float32, shape=([10, 2, 3]))
Y = tf.placeholder(tf.float32, shape=([10, 1, 2]))
W = tf.Variable(tf.random_normal([10, 3, 1]))
b = tf.Variable(tf.random_normal([10, 2, 1]))

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
cost_history=[]

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    cost_history.append(sess.run(cost, feed_dict={X: x_data, Y: y_data}))

plt.figure(figsize=[12,6])
plt.plot(cost_history)
plt.grid()
#plt.show()﻿
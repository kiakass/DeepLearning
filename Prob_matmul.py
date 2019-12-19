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

tf.shape(x_data).eval()
tf.shape(y_data).eval()

X = tf.placeholder(tf.float32, shape=([10, 2, 3]))
Y = tf.placeholder(tf.float32, shape=([10, 1, 2]))
W = tf.Variable(tf.random_normal([10, 3, 1]))
b = tf.Variable(tf.random_normal([10, 2, 1]))

'''

X와 W의 matmul 을 가능한 값을 찾아야 함

4*6 : np.dot(A[i,j,k,:], B[n,o,p,i,:,m]) => A,B[i,j,k,n,o,p,i,m]﻿
4*6 : np.matmul(A[i,j,k,:], B[n,o,p,i,j,:,m]) => A,B[n,o,p,i,j,k,m]﻿

X = tf.placeholder(tf.float32, shape=([10, 2, 3]))
Y = tf.placeholder(tf.float32, shape=([10, 1, 2]))
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

'''

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


##################
X = np.arange(10*2*3).reshape((10,2,3))
Y = np.arange(10*1*2).reshape((10,1,2))
W = np.arange(3*2).reshape((3,2))
b = np.arange(1).reshape((1))
b = np.arange(10*2*1).reshape((10,2,1))
print(X,X.shape)
print(Y,Y.shape)
print(W,W.shape)
print(b,b.shape)

H = np.matmul(X,W)
print(H,H.shape)
H1 = np.matmul(X,W) + b
print(H1,H1.shape)
H2 = Y*H1
print(H2,H2.shape)
H3 = np.dot(Y,H1)
print(H3,H3.shape)

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:37:43 2019

@author: Administrator

Numpy, Tensor 만드는 방법
변수 만드는 방법

"""

import tensorflow as tf
import numpy as np
from functions import showOperation as show
import functions
 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# np.array
t = np.array([0., 1., 2., 3., 4., 5., 6., 7.]).reshape(2,4)
print(t,t.shape)
# np.arange
arr_data = np.arange(24).reshape((2, 3, 4))
print(arr_data,arr_data.shape)
# tf.constant
x = tf.constant([0,1,2,3,4,5], shape=[2,3])
print(sess.run([x, tf.shape(x)]))
# tf.reshape
x = tf.constant([0,1,2,3,4,5])
tf.reshape(x,[2,3])
print(sess.run(tf.reshape(tf.constant([0,1,2,3,4,5]),[2,3])))

# zeros_like
x = tf.constant([[0,1,2],[3,4,5]], dtype=tf.int32)

x = np.array([0,1,2,3,4,5]).reshape(2,-1)
print(sess.run(tf.zeros_like(x)))

print(sess.run(tf.zeros_like(tf.zeros([2,2,3]))))
print(sess.run(tf.zeros([2,3])))

# ones_like
print(sess.run(tf.ones_like(x)))
print(sess.run(tf.ones([2,3])))

# fill
print(sess.run(tf.fill([2, 3], 9)))
y = tf.fill([2, 3], 9).reshape((3,2))
print(sess.run(y))

# tf.random_normal
print(sess.run(tf.random_normal([2, 3], mean=5.0, stddev=1.35)))
print(sess.run(tf.random_normal([1],1, 0.1)))

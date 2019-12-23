# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 08:46:50 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = tf.constant(np.arange(1*3*4).reshape([1,3,4]))
y = tf.constant(np.arange(5*1*4).reshape((5,1,4)))

x.shape()
x.get_shape()

z = tf.subtract(x, y)
print(sess.run([x, tf.shape(x)]),'\n',sess.run([y,tf.shape(y)]),'\n',
      sess.run([z, tf.shape(z)]))
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:24:02 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf
sess=tf.Session()

# 배열 만드는 방법
a = np.array([(100.0, 200.0),(300.0, 400.0)], dtype='float64')
a.shape
b = np.arange(2*2).reshape((2,2))
b.shape
b.ndim
print(b)

vectors = tf.constant(b)
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_vectors = tf.expand_dims(vectors, 1)
expanded_vectors = tf.expand_dims(vectors, 2)
expanded_vectors = tf.expand_dims(vectors, 3)
# shape를 쓰는 방법
expanded_vectors.get_shape()

with tf.Session() as sess:
    print(sess.run(tf.shape(expanded_vectors)))

sess=tf.Session()
print(sess.run(tf.shape(expanded_vectors)))
print(sess.run(expanded_vectors))


c = tf.zeros_like
print(sess.run(c))

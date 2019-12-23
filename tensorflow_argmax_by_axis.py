# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:46:30 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = tf.constant(np.arange(1*3*4).reshape([1,3,4]))
print(sess.run(x),'\n\n',sess.run(a))


x = tf.random_shuffle(tf.constant(np.arange(10*4).reshape([10,4])))
print(sess.run([x]))

a = tf.argmin(x,1)

print(sess.run(x),'\n\n',sess.run(a))

b = tf.argmax(x,1)
print(sess.run(x),'\n\n',sess.run(b))


x = (tf.constant(
        np.array([0,1,2,8,4,5,6,7,11,9,10,3]).reshape([3,4])))
a = tf.argmin(x,1)
print(sess.run(x),'\n\n',sess.run(a))
b = tf.argmax(x,1)
print(sess.run(x),'\n\n',sess.run(b))


# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 08:28:53 2019

@author: Administrator
"""
import numpy as np
import tensorflow as tf

a=np.array([(100.0, 200.0),(300.0, 400.0)], dtype='float64')
b=np.array([(2.0, 3.0), (7.0, 11.0)], dtype='float64')
c=np.array([(0.11, -0.55), (-6.6, 12.9)], dtype='float64')
d=np.array([[1,2,3],[2,3,5],[1,0,2]], dtype='float64')
print(a,'\n\n',b,'\n\n',c)

ta=tf.convert_to_tensor(a, dtype=tf.float64)
tb=tf.convert_to_tensor(b, dtype=tf.float64)
tc=tf.convert_to_tensor(c, dtype=tf.float64)
td=tf.convert_to_tensor(d, dtype=tf.float64)

print(sess.run(ta),'\n\n', sess.run(tb),'\n\n', sess.run(tc))

sess=tf.Session()

# 역함수
sess.run(ta)
sess.run(tf.matrix_inverse(ta))
sess.run(tf.matrix_inverse(td))

# 스칼라곱, dot, matmul
print(sess.run(ta),'\n\n', sess.run(tb),'\n\n', sess.run(tc))

sess.run(ta*tb)
sess.run(np.dot(ta,tb))
sess.run(tf.matmul(ta, tb))

# 역함수 matmul
sess.run(tf.matmul(ta,tf.matrix_inverse(ta)))


# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:31:56 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

''' ---------------------------------------------------------------- '''
t = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
pp.pprint(t)
print(t)
print(t[0],t[1],t[-1])
print(t[2:5])

#2D Array

t = np.array([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
pp.pprint(t)
print('array : \n',t)
print('dim :', t.ndim,'\nshape :',t.shape)
tf.shape(t).eval()


#matmul vs multiply

m1 = tf.constant([[1.,2.],[3.,4.]])
m2 = tf.constant([[1.],[2.]])
print("m1 shape : ", m1)
print("m2 shape : ", m2)

tf.matmul(m1,m2).eval()
(m1*m2).eval()

'''
matrix multiply(matmul) 는 
[1.,2.] * [1.] = [5.]  (=1.*1.+2.*2.)
[3.,4.]   [2.]   [11.] (=3.*1.+4.*2.)

multiply 는 element 끼리 곱하고, Broadcasting 이 일어남
[1.,2.] * [1.,1.] = [1.,2.] (=1.*1.,=2.*1.)
[3.,4.]   [2.,2.]   [6.,8.] (=3.*2.,=4.*2.)

'''

# Broadcating
m1 = tf.constant([[1.,2.]])
m2 = tf.constant([[3.],[4.]])

(m1+m2).eval()
'''
Broadcasting이 일어나고, element끼리 더해짐
[[1.,2.],[1.,2.]] + [[3.,3.],[4.,4.]] = [[4.,5.],[5.,6.]]

'''

# Reduce mean
tf.reduce_mean([1,2], axis=0).eval()

x = [           # axis = 0 
     [1., 2.],  # axis = 1
     [3., 4.]
    ]

tf.reduce_mean(x).eval()
tf.reduce_mean(x, axis=0).eval()
tf.reduce_mean(x, axis=1).eval()

# Reduce sum

tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()


# Argmax
x = [[0,1,2],[2,1,0]]
tf.argmax(x).eval()  # default : axis = 0

tf.argmax(x, axis=0).eval()
tf.argmax(x, axis=1).eval()
tf.argmax(x, axis=-1).eval()


# squeeze 차원중 사이즈 1인 것을 찾아 제거한다.
tf.squeeze([[0],[1],[2]]).eval(session=sess)
tf.squeeze([[[1],[2]],[[3],[4]]]).eval()
tf.squeeze([[0,1,2],[2,1,0]]).eval()

# expand_dims 함수
x = np.array([0,1,2])
print("x.shape: ", x.shape)
tf.expand_dims(x,0).eval()
tf.expand_dims(x,1).eval()
tf.expand_dims(x,2).eval()

x = np.array([[1,2],[3,4]])
x = np.array([[[1,2],[3,4]]])
x = np.array([[[1],[2],[3]]])
print("x.shape: ", x.shape)
tf.expand_dims(x,0).eval()
y = tf.expand_dims(x,0).eval()
print(y.shape)
tf.expand_dims(x,1).eval()
y = tf.expand_dims(x,1).eval()
print(y.shape)
tf.expand_dims(x,2).eval()
y = tf.expand_dims(x,2).eval()
print(y.shape)
tf.expand_dims(x,3).eval()
tf.expand_dims(x,4).eval()

arr_data = np.arange(24).reshape((2, 3, 4))
print(arr_data,arr_data.shape)


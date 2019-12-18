# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:52:45 2019

@author: Administrator
"""
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

arr_data1 = np.arange(2*10*3).reshape((10, 2, 3))
print(arr_data1,arr_data1.shape)


arr_data2 = np.arange(10*3*1).reshape((10,3,1))
print(arr_data2,arr_data2.shape)
'''
arr_data2 = np.arange(2*3*4).reshape((3, 4, 2))
print(arr_data2,arr_data2.shape)
'''
t = tf.matmul(arr_data1,arr_data2).eval()
print(t, t.shape)

# Dot 연산
import numpy as np
A = np.arange(2*3*4).reshape((2,3,4))
B1 = np.arange(2*3*4).reshape((2,3,4))
B2 = np.arange(2*3*4).reshape((2,4,3))
B3 = np.arange(2*3*4).reshape((3,2,4))
B4 = np.arange(2*3*4).reshape((3,4,2))
B5 = np.arange(2*3*4).reshape((4,2,3))
B6 = np.arange(2*3*4).reshape((4,3,2))

C2_dot = np.dot(A,B2)
print("A :\n",A,A.shape,"\n\nB2 :\n",B2,B2.shape,"\n\nC2.dot :\n",C2_dot,C2_dot.shape,)

C2_mat = np.matmul(A,B2)
print("A :\n",A,A.shape,"\n\nB2 :\n",B2,B2.shape,"\n\nC2.matmul :\n",C2_mat,C2_mat.shape,)


A = np.arange(2*3*4*4).reshape((2,3,4,4))
B = np.arange(2*2*6*4*5).reshape((2,2,6,4,5))
C = np.arange(1*2*3*4*5*6).reshape((1,5,2,3,4,6))


AB_dot = np.dot(A,B)
print("A :\n",A,A.shape,"\n\nB :\n",B,B.shape,"\n\nAB.dot :\n",AB_dot,AB_dot.shape,)

AC_dot = np.dot(A,C)
print("A :\n",A,A.shape,"\n\nB :\n",C,C.shape,"\n\nAB.dot :\n",AC_dot,AC_dot.shape,)


#matmul
A = np.arange(3*2*3*4*5).reshape((3,2,3,4,5))
B = np.arange(3*3*2*3*4*5).reshape((3,3,2,3,5,4))
C = np.arange(2*3*2*3*5*6).reshape((2,3,2,3,5,6))

AB_matmul = np.matmul(A,B)
print("A :\n",A,A.shape,"\n\nB :\n",B,B.shape,"\n\nAB.matmul :\n",AB_matmul,AB_matmul.shape,)

AC_matmul = np.matmul(A,C)
print("A :\n",A,A.shape,"\n\nB :\n",C,C.shape,"\n\nAB.matmul :\n",AC_matmul,AC_matmul.shape,)

'''
https://blog.naver.com/wideeyed/221047109563
'''
A = np.arange(2*3*4).reshape((2,3,4))
print(A)
t = tf.transpose(A)
print(t)

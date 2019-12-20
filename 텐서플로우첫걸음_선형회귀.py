# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:27:58 2019

@author: junhk
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
num_points = 1000
vectors_set = []

'''
np.random.normal(mean,std,size)
±3σ 범위에 대부분 존재함 
(0.0, 0.55) -> -1.75< <1.75
'''
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.3)
    #print(x1)
   # print(y1)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
plt.plot(x_data, y_data, 'ro')
plt.show()
# -1< <1 사이의 값을 가짐
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# '0'으로 된 list를 만듬, b = np.zeros(10)
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b 

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20):
    _, W_val, b_val,loss_val = sess.run([train, W, b, loss])
    print('step : ',step, 'loss : ',loss_val, 'W : ',W_val, 'b : ',b_val)


plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W)*x_data + sess.run(b))
plt.xlabel('x')
plt.xlim(-2,2)
plt.ylim(0.1,0.6)
plt.ylabel('y')
plt.show()









strings = ['gorgeous','sick','sad','happy','joyful']
print(enumerate(strings))
strings

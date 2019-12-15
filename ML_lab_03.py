# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:29:49 2019

@author: Administrator
"""

import tensorflow as tf

import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
# hypothsis for linear model X * W
hypothesis = X * W

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
# Variables for plotting cost function
W_val=[] #그래프를 그리기위한 변수
cost_val=[]
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)
    
plt.plot(W_val, cost_val)
plt.show()

## 2. 미분식으로 구함
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W
cost = tf.reduce_sum(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W*X - Y)*X)
# Descent 함수 Convex 구로조 인해 W가 1이면 gradient 가 0이되는 구조임
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    # placeholder 를 쓰면 feed_dict를 통해 값을 받아옴
    update_val, cost_val, W_val = sess.run([update,cost,W], feed_dict={X: x_data, Y: y_data})
    print(step, cost_val, update_val, W_val)
#    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    
## 3.Optimizer를 사용함
    
X = [1, 2, 3]
Y = [1, 2, 3]

#W = tf.Variable(-3.0)
W = tf.Variable(tf.random_normal([1]), name='weight')
hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

############
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    _, W_val = sess.run([train, W])
    print(step, W_val)
    print(step, sess.run(W))
    sess.run(train)

#############
# Minimize: Gradient Descent Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _,Cost_val, W_val = sess.run([train,cost, W])
        print(step, W_val, Cost_val)

# Lab 3 Minimizing Cost
# This is optional
import tensorflow as tf

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.)
# Linear model
hypothesis = X * W
# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize: Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Get gradients
gvs = optimizer.compute_gradients(cost, W)
# Apply gradients = minimize
apply_gradients = optimizer.apply_gradients(gvs)
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
'''
# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        gradient_val, gvs_val, _ = sess.run([gradient, gvs, apply_gradients])
        print(step, gradient_val, gvs_val)
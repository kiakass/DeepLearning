# -*- coding: utf-8 -*-
"""
XOR Using Logistic Regression
"""
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = - tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# cast True:1, False:0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# Training, learning_rate 을 0.1씩 옮기면서 경사하강을 하면서 W,b를 10000번 업데이트함    
    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

# Validation            
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print('\nhypothesis: ',h,'\nCorrect: ',c,'\nAccuracy: ',a)

"""
XOR Using Neural Network
"""

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 층(Layer)를 쌓음, Wide하게, Depth있게
W1 = tf.Variable(tf.random_normal([2,10]), name='weight')
b1 = tf.Variable(tf.random_normal([10]), name='bias')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10,10]), name='weight')
b2 = tf.Variable(tf.random_normal([10]), name='bias')
layer2 = tf.sigmoid(tf.matmul(X, W1) + b1)

W3 = tf.Variable(tf.random_normal([10,1]), name='weight')
b3 = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)

cost = - tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# cast True:1, False:0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# Training, learning_rate 을 0.1씩 옮기면서 경사하강을 하면서 W,b를 10000번 업데이트함    
    for step in range(100001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 10000 == 0:
            print( step, sess.run([cost], feed_dict={X: x_data, Y: y_data}) )
    # Validation            
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print('\nhypothesis: ',h,'\nCorrect: ',c,'\nAccuracy: ',a)

'''            cost_val,  W3_val,  b3_val = sess.run([cost, W3, b3],
                                                       feed_dict={X: x_data, Y: y_data})
            print("cost : ", cost_val,"W3 : ", W3_val, "b3 : ", b3_val)
'''


# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:39:46 2019

@author: Administrator

hidden layer 를 늘이면 epoch도 늘여야함

"""

# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("d:/data/mnist", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784], name="x")
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes], name="y")

### 1 Decide which tensors you want to log
#W = tf.Variable(tf.random_normal([784, nb_classes]))
#b = tf.Variable(tf.random_normal([nb_classes]))
# 층(Layer)를 쌓음, Wide하게, Depth있게
W1 = tf.Variable(tf.random_normal([784,100]), name='weight')
b1 = tf.Variable(tf.random_normal([100]), name='bias')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W1_hist = tf.summary.histogram("weights1", W1)
b1_hist = tf.summary.histogram("biases1", b1)
layer1_hist = tf.summary.histogram("layer1", layer1)

W2 = tf.Variable(tf.random_normal([100,10]), name='weight')
b2 = tf.Variable(tf.random_normal([10]), name='bias')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W2_hist = tf.summary.histogram("weights2", W2)
b2_hist = tf.summary.histogram("biases2", b2)
layer2_hist = tf.summary.histogram("layer2", layer2)

W3 = tf.Variable(tf.random_normal([10,nb_classes]), name='weight')
b3 = tf.Variable(tf.random_normal([nb_classes]), name='bias')
#hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)
# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

W3_hist = tf.summary.histogram("weights3", W3)
b3_hist = tf.summary.histogram("biases3", b3)
hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

### 2 Summary
summary = tf.summary.merge_all()

### 3 initialize

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 30
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)
global_step = 0
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    writer = tf.summary.FileWriter('d:/data/logs')
    writer.add_graph(sess.graph)
    
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val, s = sess.run([train, cost, summary], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations
            
            writer.add_summary(s, global_step=global_step)
            global_step += 1

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )
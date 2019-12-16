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
mnist = input_data.read_data_sets("d:/data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

#W = tf.Variable(tf.random_normal([784, nb_classes]))
#b = tf.Variable(tf.random_normal([nb_classes]))
# 층(Layer)를 쌓음, Wide하게, Depth있게
W1 = tf.Variable(tf.random_normal([784,100]), name='weight')
b1 = tf.Variable(tf.random_normal([100]), name='bias')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([100,10]), name='weight')
b2 = tf.Variable(tf.random_normal([10]), name='bias')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
'''
W3 = tf.Variable(tf.random_normal([10,100]), name='weight')
b3 = tf.Variable(tf.random_normal([100]), name='bias')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([100,10]), name='weight')
b4 = tf.Variable(tf.random_normal([10]), name='bias')
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([10,nb_classes]), name='weight')
b5 = tf.Variable(tf.random_normal([nb_classes]), name='bias') '''
W3 = tf.Variable(tf.random_normal([10,nb_classes]), name='weight')
b3 = tf.Variable(tf.random_normal([nb_classes]), name='bias')
#hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)
# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

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

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:58:27 2019

@author: Administrator
"""
import tensorflow as tf
tf.global_variables_initializer()

a = tf.random_uniform([1])
b = tf.random_normal([1])

print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A3'
  print(sess2.run(a))  # generates 'A4'
  print(sess2.run(b))  # generates 'B3'
  print(sess2.run(b))  # generates 'B4'
  
a = tf.random_uniform([1], seed=1) # seed를 주고 random한 값을 생성하면 매 session마다 값이 같음
b = tf.random_normal([1])

print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A1'
  print(sess2.run(a))  # generates 'A2'
  print(sess2.run(b))  # generates 'B3'
  print(sess2.run(b))  # generates 'B4'

# tf.set_random_seed를 통해 모든 random value generation function들이 매번 같은 값을 반환함    
tf.set_random_seed(1234)
a = tf.random_uniform([1])
b = tf.random_normal([1])

# Repeatedly running this block with the same graph will generate different
# sequences of 'a' and 'b'.
print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A1'
  print(sess2.run(a))  # generates 'A2'
  print(sess2.run(b))  # generates 'B1'
  print(sess2.run(b))  # generates 'B2'
  
# Create a tensor of shape [2, 3] consisting of random normal values, with mean
# -1 and standard deviation 4.
norm = tf.random_normal([2, 3], mean=-1, stddev=4)

# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

# Each time we run these ops, different results are generated
sess = tf.Session()
print(sess.run(shuff))
print(sess.run(norm))
print(sess.run(norm))

norm = tf.random_normal([2, 3], seed=1234)
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))

sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))

t = ([[[1, 1, 1], [2, 2, 2]],
      [[3, 3, 3], [4, 4, 4]],
      [[5, 5, 5], [6, 6, 6]]])
t[0,1,1]
t[1,1,0]
    
 # [[[3, 3, 3]]]
print(sess.run(tf.slice(t, [0, 0, 0], [0, 0, 0])))  # [[[3, 3, 3],
print(sess.run(tf.slice(t, [0, 0, 0], [0, 0, 0])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 0, 0])))
print(sess.run(tf.slice(t, [0, 0, 0], [0, 1, 0])))
print(sess.run(tf.slice(t, [0, 0, 0], [0, 0, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 1, 0])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 0, 1])))
print(sess.run(tf.slice(t, [1, 0, 0], [1, 1, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [2, 1, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [3, 1, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [3, 2, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [3, 2, 2])))
print(sess.run(tf.slice(t, [0, 0, 0], [3, 2, 3])))
                                   #   [4, 4, 4]]]
print(sess.run(tf.slice(t, [0, 0, 0], [1, 1, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 2, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 3, 1])))

print(sess.run(tf.slice(t, [0, 0, 0], [1, 1, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 1, 2])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 1, 3])))
 # [[[3, 3, 3]],
print(sess.run(tf.slice(t, [0, 0, 0], [1, 2, 1])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 2, 2])))
print(sess.run(tf.slice(t, [0, 0, 0], [1, 2, 3])))
print(sess.run(tf.slice(t, [0, 0, 0], [0, 2, 3])))
print(sess.run(tf.slice(t, [1, 1, 3], [3, 2, 3])))

print(sess.run(tf.slice(t, [0, 0, 0], [1, 1, 1])))
print(sess.run(tf.slice(t, [1, 0, 0], [1, 1, 1])))
print(sess.run(tf.slice(t, [2, 0, 0], [1, 1, 1])))

print(sess.run(tf.slice(t, [1, 0, 0], [1, 1, 1])))
print(sess.run(tf.slice(t, [1, 1, 0], [1, 1, 1])))
print(sess.run(tf.slice(t, [1, 1, 1], [1, 1, 1])))
print(sess.run(tf.slice(t, [1, 1, 2], [1, 1, 1])))
print(sess.run(tf.slice(t, [1, 1, 1], [1, 1, 2])))
print(sess.run(tf.slice(t, [1, 1, 0], [1, 1, 3])))
print(sess.run(tf.slice(t, [1, 1, 1], [1, 1, 2])))
print(sess.run(tf.slice(t, [1, 1, 1], [2, 1, 2])))
print(sess.run(tf.slice(t, [1, 1, 2], [1, 1, 2])))
print(sess.run(tf.slice(t, [1, 0, 0], [2, 2, 2])))
print(sess.run(tf.slice(t, [1, 1, 1], [1, 1, 3])))
print(sess.run(tf.slice(t, [1, 1, 2], [2, 1, 1])))
print(sess.run(tf.slice(t, [1, 0, 0], [2, 2, 2])))
print(sess.run(tf.slice(t, [1, 0, 1], [2, 2, 2])))



x = tf.constant([[[1., 2.], [3., 4. ], [5. , 6. ]],
                 [[7., 8.], [9., 10.], [11., 12.]]])

# Extracts x[0, 1:2, :] == [[[ 3.,  4.]]]
print(sess.run(tf.slice(x, [0, 1, 0], [1, 1, -1])))

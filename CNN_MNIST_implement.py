#coding=utf-8
"""
This program using tensorflow to apply CNN to the MNIST dataset.

__author__ = "Tiancong Li"
"""
import tensorflow as tf

print("start loading MNIST dataset.")
from tensorflow.examples.tutorials.mnist import input_data

STEPS = 101
BATCH_SIZE = 50

# load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create placeholders
X = tf.placeholder(tf.float32, [None, 28*28])
y_true = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# create first convolutional layer
X_shaped = tf.reshape(X, shape=[-1, 28, 28, 1])
conv1 = tf.layers.conv2d(X_shaped, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)

# create first pooling layer
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

# create second convolutional layer
conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)

# create second pooling layer
pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

# create first dense layer
pool2_shaped = tf.reshape(pool2, shape=[-1, 7*7*64])
dense = tf.layers.dense(pool2_shaped, units=1024, activation=tf.nn.relu)

# create dropout
dropout = tf.nn.dropout(dense, keep_prob=keep_prob)

# create output layer
logits = tf.layers.dense(dropout, units=10)

# create loss function and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

y_equals = tf.equal(tf.argmax(y_true, 1), tf.argmax(logits, 1))
acc = tf.reduce_mean(tf.cast(y_equals, tf.float32))

# train and evaluate the model, print the accuracy
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(STEPS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train, feed_dict={X:batch_x, y_true:batch_y, keep_prob:0.5})

        if i % 100 == 0:
            print(sess.run(acc, feed_dict={X:mnist.test.images, y_true:mnist.test.labels, keep_prob:1.0}))
    




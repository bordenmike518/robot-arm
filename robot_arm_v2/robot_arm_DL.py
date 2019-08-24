''' Demonstrates deep learning using fully connected layers and tuning '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as learn
from tensorflow.contrib.layers import fully_connected

# Read in training data
position_train = np.load('data/position_train_120k.npy').astype('float32')
joint_train = np.load('data/joint_train_120k.npy').astype('float32')
assert (position_train.shape[0] == joint_train.shape[0])
position_train_placeholder = tf.placeholder(dtype=tf.float32, position_train.shape)
joint_train_placeholder = tf.placeholder(dtype=tf.float32, joint_train.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((position_train_placeholder, joint_train_placeholder))
# Read in testing data
position_test = np.load('data/position_test_29k.npy').astype('float32')
joint_test = np.load('data/joint_test_29k.npy').astype('float32')
assert (position_test.shape[0] == joint_test.shape[0])
position_test_placeholder = tf.placeholder(dtype=tf.float32, position_test.shape)
joint_test_placeholder = tf.placeholder(dtype=tf.float32, joint_test.shape)
test_dataset = tf.data.Dataset.from_tensor_slices((position_test_placeholder, joint_test_placeholder))

# Create a placeholder
train = tf.placeholder(tf.bool)

# Layer settings
hid_nodes = 200
out_nodes = 6
keep_prob = 0.5

# Create layers
with tf.contrib.framework.arg_scope(
    [fully_connected],
    normalizer_fn=tf.contrib.layers.batch_norm,
    normalizer_params={'is_training': train}):
        layer1 = fully_connected(position_train_placeholder, hid_nodes, scope='layer1')
        layer1_drop = tf.layers.dropout(layer1, keep_prob, training=train)
        layer2 = fully_connected(layer1_drop, hid_nodes, scope='layer2')
        layer2_drop = tf.layers.dropout(layer2, keep_prob, training=train)
        layer3 = fully_connected(layer2_drop, hid_nodes, scope='layer3')
        layer3_drop = tf.layers.dropout(layer3, keep_prob, training=train)
        out_layer = fully_connected(layer3_drop, out_nodes,
            activation_fn=None, scope='layer4')

# Compute loss
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=out_layer, labels=joint_train_placeholder))

# Create optimizer
learning_rate = 0.01
num_epochs = 15
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Launch session
with tf.Session() as sess:
    sess.run(init)

    # Loop over epochs
    for epoch in range(num_epochs):
        sess.run(optimizer, feed_dict={position_train_placeholder: position_train,
            joint_train_placeholder: joint_train, train: True})

    # Determine success rate
    prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(joint_test_placeholder, 1))
    success = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print('Success rate: ', sess.run(success,
           feed_dict={position_test_placeholder: position_test,
                      joint_test_placeholder: joint_test, train: False}))

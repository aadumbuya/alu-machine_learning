#!/usr/bin/env python3

"""placeholders"""


import tensorflow as tf

def create_placeholders(nx, classes):
    """
    Creates two placeholders, x and y, for the neural network.

    Arguments:
    nx -- number of feature columns in our data
    classes -- number of classes in our classifier

    Returns:
x -- placeholder for the input data, of shape [None, nx] and dtype "float"
    y -- placeholder for the one-hot labels, of shape [None, classes] and dtype "float"
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    
    return x, y





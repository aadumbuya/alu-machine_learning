#!/usr/bin/env python3
"""Evaluate the model"""


import tensorflow as tf
import numpy as np

def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Arguments:
    X -- numpy.ndarray containing the input data to evaluate
    Y -- numpy.ndarray containing the one-hot labels for X
    save_path -- location to load the model from

    Returns:
Prediction, accuracy and loss of the network
    """"
    with tf.Session() as sess:
        # Load the saved model
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
We retrieve the tensors from the collection in the graph
        graph = tf.get_default_graph()
        x     = graph.get_collection('x')    [0]
        y     = graph.get_collection('y')    [0]
        y_pred= graph.get_collection('y_pred')[0]
        loss  = graph.get_collection('loss') [0]
accuracy = tf.get_collection('accuracy')[0]
        # Model evaluation
        predictions = sess.run(y_pred, feed_dict={x: X, y: Y})
        loss_value = sess.run(loss, feed_dict={x: X, y: Y})
accuracy_value = sess.run(accuracy, feed_dict={x: X, y: Y})
        #
    return predictions, accuracy_value, loss_value



#!/usr/bin/env python3
"""Forward Propagation"""



import tensorflow as tf 

create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Arguments:
    x -- placeholder for the input data
    layer_sizes -- list containing the number of nodes in each layer of the network
activations -- list of the activation functions for each layer of the network

    Returns:
    The prediction of the network in tensor form
    """
"""
    layer = x
    for size, activation in zip(layer_sizes, activations):
        layer = create_layer(layer, size, activation)
    return layer

    

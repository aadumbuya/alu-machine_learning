#!/usr/bin/env python3

import numpy as np

def l2_reg_cost(cost, lambtha, weights, l, m):
    """
    computes cost of neural network with l2 regularization

    Parameters:
    - cost : cost of the net without l2 regularization
    - lambtha : regularization parameter
    - weights: dict of the weights and biases (numpy.ndarrays) of the neural network
- l: number of layers in the neural network
    - m: number of data points used
    ''
    Returns:
    - The cost of the network accounting for L2 regularization
    """
    l2_cost = cost
    l2_sum =
for i in range(1, l + 1):
        weight_key = 'W' + str(i)
        l2_sum += np.sum(np.square(weights[weight_key]))
    
    l2_cost += (lambtha / (2 * m)) * l2_sum
    
    return l2_cost


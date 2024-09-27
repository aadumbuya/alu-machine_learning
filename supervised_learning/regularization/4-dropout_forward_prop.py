#!/usr/bin/env python 3 

import numpy as np

def dropout_forward_prop(X, parameters, L, keep_prob):
    cache = {'A0': X}

    for i in range(1, L + 1):
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
Z = np.dot(W, A_prev) + b
        #BACKWARD PASS
        if i == L:
            # Softmax activation in the last layer
t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Tanh activation for hidden layers
A = np.tanh(Z)
            # Dropout mask
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D)
A /= keep_prob
            cache['D' + str(i)] = D
        # 
        cache['A' + str(i)] = A
    # 
    return cache


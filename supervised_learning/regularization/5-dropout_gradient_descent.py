#!/usr/bin/env python3

import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
b = weights['b' + str(i)]

        dW = (1 / m) * np.dot(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
dz = np.dot(W.T, dz) * (1 - A_prev ** 2)
            dz *= cache['D' + str(i - 1)]
            dz /= keep_prob

        weights['W' + str(i)] -= alpha * dW
weights['b' + str(i)] -= alpha * db


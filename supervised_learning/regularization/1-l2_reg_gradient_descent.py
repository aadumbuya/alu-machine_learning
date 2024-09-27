#!/usr/bin/env python3

Import numpy as np

def l2_reg_gradient_descent(y, weights, cache, alpha, lambtha, l):
    m = y.shape[1]
    dz = cache['A' + str(l)] - y

    for i in range(l, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
b = weights['b' + str(i)]
        #update rule 
        dW = (1 / m) * np.dot(dz, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
if i > 1:
            dz = np.dot(W.T, dz) * (1 - A_prev ** 2)
        #    
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db


#!/usr/bin/env python3
""" RMSProp"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Function that updates a variable using the
    RMSProp optimization algorithm
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s

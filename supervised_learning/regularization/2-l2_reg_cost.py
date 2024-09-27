#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def l2_reg_cost(cost, weights, lambtha):
    l2_cost = cost
    l2_regularization = 0
    for weight in weights.values():
        l2_regularization += tf.nn.l2_loss(weight)
    l2_cost += lambtha * l2_regularization
    return l2_cost


#!/bin/usr/env python3

import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )
return layer(prev=



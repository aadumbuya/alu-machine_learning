#!/usr/bin/env python3

import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob):
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )(prev)

    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)(layer)
Dropout end


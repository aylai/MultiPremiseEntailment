import math

import tensorflow as tf
import numpy as np

tf.set_random_seed(20170302)

"""
Helper class for quickly creating NN layers with different initializations
"""


def ff_w(input_dim=100, output_dim=100, name="W", init="Xavier", reg=None, reg_val=None):
    return variable([input_dim, output_dim], name, init, reg, reg_val)


def ff_b(dim=100, name="B", init="Zero", reg=None):
    return variable([dim], name, init, reg)


def variable(shape, name, init="Xavier", reg=None, reg_val=None, trainable=True):
    with tf.variable_scope(name) as scope:
        regularizer = None
        if reg == "l2" and reg_val is not None:
            regularizer = tf.contrib.layers.l2_regularizer(reg_val)
        elif reg == "l1" and reg_val is not None:
            regularizer = tf.contrib.layers.l1_regularizer(reg_val)

        if init == "Zero":
            vals = np.zeros(shape).astype("f")
            return tf.get_variable(name=name, initializer=vals, dtype=tf.float32, trainable=trainable)
        if init == "One":
            vals = np.ones(shape).astype("f")
            return tf.get_variable(name=name, initializer=vals, dtype=tf.float32, trainable=trainable)

        if init == "Xavier":
            w_init = tf.contrib.layers.xavier_initializer(seed=20161016)
        elif init == "Normal":
            w_init = tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[0]),
                                                  seed=12132015)
        elif init == "Uniform":
            w_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05,
                                                   seed=12132015)
        return tf.get_variable(name=name, shape=shape, initializer=w_init, regularizer=regularizer, trainable=trainable)

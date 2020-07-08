# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:04:19 2020
test.py
~~~~~~~~~~
Compares the standard backprop output with the fully matrix based backprop
~~~~~~~~~~
@author: Ian
"""

import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
epochs = 30
batch_size = 10
eta = 3.0

if test_data: n_test = len(test_data)
    n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
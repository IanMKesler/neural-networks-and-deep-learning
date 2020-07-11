# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:57:13 2020
learn.py
~~~~~~~~~~
Loads mnist data and performs gradient descent.
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
matrix = False
net.SGD(training_data, epochs, batch_size, eta, matrix, test_data=test_data)

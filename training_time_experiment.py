# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:53:39 2017

@author: brucew
"""

import keras
from keras.callbacks import CSVLogger
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import sys
import random
import pandas as pd
import numpy as np

from train import Train
from network import Network

import os

file_path = "./results/training_times/"
directory = os.path.dirname(file_path)

try:
    os.stat(directory)
except:
    os.mkdir(directory)       



train = Train({'dataset':'cifar10'})

for network_size in range(1, 5):

    for x in range(10):
    
        log_file_name = 'results/training_times/training_layers_%d_%d' % (network_size, x) + '.log'
    
        csv_logger = CSVLogger(log_file_name)
        
        network = Network()
        network.create_random_network(network_size)
        
        model = train.compile_model(network, train.output_shape, train.input_shape, train.natural_input_shape)
        train.train_model(model, train.x_train, train.y_train, train.batch_size, 100, train.x_test, train.y_test, [csv_logger])
    
    
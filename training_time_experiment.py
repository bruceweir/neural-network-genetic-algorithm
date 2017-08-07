# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:53:39 2017

@author: brucew
"""

import keras
from keras.callbacks import CSVLogger
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import sys
import random
import pandas as pd
import numpy as np

from train import Train
from network import Network


train = Train({'dataset':'mnist'})

for network_size in range(1, 5):

    for x in range(10):
    
        log_file_name = 'training_layers_%d_%d' % (network_size, x) + '.log'
    
        csv_logger = CSVLogger(log_file_name)
        
        network = Network()
        network.create_random_network(network_size)
        
        model = train.compile_model(network, train.output_shape, train.input_shape, train.natural_input_shape)
        train.train_model(model, train.x_train, train.y_train, train.batch_size, 100, train.x_test, train.y_test, [csv_logger])
    
    
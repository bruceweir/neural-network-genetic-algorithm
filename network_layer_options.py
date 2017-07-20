# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 21:35:05 2017

@author: brucew
"""

def get_dense_layer_options():

    return {
            'nb_neurons': [64, 128, 256, 512, 768, 1024],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid']           
    }
    
def get_reshape_layer_options():
    
    return { 
            'first_dimension_scale': [1, 2, 4, 8, 16, 32] 
    }

def get_conv2d_layer_options():
    
    return {
            'strides': [(1, 1), (2, 2), (4, 4)],
            'kernel_size': [(1, 1), (3, 3), (5, 5), (7, 7)],
            'nb_filters': [2, 8, 16, 32, 64],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid']
    }

def get_dropout_layer_options():
    
    return {
            'remove_probability':[.3, .2, .1]
    }

def get_maxpooling2d_layer_options():
    
    return {
            'pool_size': [(2, 2), (4, 4), (6, 6)]
    }


def get_layer_types_for_random_selection():
    
    return ['Dense', 'Conv2D', 'MaxPooling2D', 'Dropout']
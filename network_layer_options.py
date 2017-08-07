# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 21:35:05 2017

@author: bruce.weir@bbc.co.uk
"""

def get_dense_layer_options():

    """Returns the optional parameters and values for a Dense (Fully connected) layer """

    return {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid']       
    }


def get_conv2d_layer_options():

    """Returns the optional parameters and values for a 2D convolution layer """

    return {
        'strides': [(1, 1), (2, 2), (4, 4)],
        'kernel_size': [(1, 1), (3, 3), (5, 5), (7, 7)],
        'nb_filters': [2, 4, 8, 16, 32],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid']
    }


def get_dropout_layer_options():

    """Returns the optional parameters and values for a Dropout layer """

    return {
        'remove_probability':[.3, .2, .1]
    }


def get_maxpooling2d_layer_options():

    """Returns the optional parameters and values for a 2D MaxPooling layer """

    return {
        'pool_size': [(2, 2), (4, 4), (6, 6)]
    }
  
    
def get_activity_regularization_layer_options():
    
    return { 
        'l1': [0.001, 0.005, 0.01, 0.05],
        'l2': [0.001, 0.005, 0.01, 0.05]
    }


def get_layer_types_for_random_selection():

    """ Returns the layer names that can be used when choosing a layer type to add """

    return ['Dense', 'Conv2D', 'MaxPooling2D', 'Dropout', 'ActivityRegularization']

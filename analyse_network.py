# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:14:34 2017

@author: bruce.weir@bbc.co.uk
"""

from keras.models import Sequential
import matplotlib.pyplot as plt


def get_activations_from_model_layer(model, layer_index, input_batch):
    
    activation_model = Sequential()
    
    for x in range(layer_index):
        activation_model.add(model.layers[x])
    
    activations = activation_model.predict(input_batch)
    
    return activations


def display_activations(activations):
    
    n_activations = activations.shape[0]
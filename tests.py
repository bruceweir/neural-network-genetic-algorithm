# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:38:41 2017

@author: brucew
"""

from network import Network
from optimizer import Optimizer
import train

def test_network():
    
    print('network.add_layer_with_random_parameters("layer_type") should add a network layer of the requested layer_type')
    
    network=Network()
    
    network.add_layer_with_random_parameters('Dense')
    
    assert(network.network_layers[0]['layer_type'] == 'Dense')
    
    
    print('network.check_network_structure() should alter a network to make it conform to certain rules')
    print('\t1. Insert a Flatten() layer if going from a 2D layer to a Dense layer.')
    network = Network()
    network.add_layer_with_random_parameters('Conv2D')
    network.add_layer_with_random_parameters('Dense')
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Conv2D')
    assert(network.get_network_layer_type(1) == 'Flatten')
    assert(network.get_network_layer_type(2) == 'Dense')
    
    print('\t2. Dropout layers cannot immediately follow Dropout layers.')
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.add_layer_with_random_parameters('Dropout')
    network.add_layer_with_random_parameters('Dropout')
    network.add_layer_with_random_parameters('Dense')
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Dense')
    assert(network.get_network_layer_type(1) == 'Dropout')
    assert(network.get_network_layer_type(2) == 'Dense')
    
    
test_network()
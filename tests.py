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
    
    print("\t3. A Reshape layer should be added between a 2D layer and a 1D layer")
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.add_layer_with_random_parameters('Conv2D');
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Dense')
    assert(network.get_network_layer_type(1) == 'Reshape')
    assert(network.get_network_layer_type(2) == 'Conv2D')
    
    print("\t4. A Reshape cannot be called between 2 2D layers")
    network = Network()
    network.add_layer_with_random_parameters('Conv2D')
    network.add_layer_with_random_parameters('Reshape')
    network.add_layer_with_random_parameters('Conv2D')
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Conv2D')
    assert(network.get_network_layer_type(1) == 'Conv2D')
    
    print('\tDropout layers should be handled correctly, depending if they are following a 1d or 2d layer')
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.add_layer_with_random_parameters('Dropout')
    network.add_layer_with_random_parameters('Conv2D')
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Dense')
    assert(network.get_network_layer_type(1) == 'Dropout')
    assert(network.get_network_layer_type(2) == 'Reshape')
    assert(network.get_network_layer_type(3) == 'Conv2D')
    
    network = Network()
    network.add_layer_with_random_parameters('Conv2D')
    network.add_layer_with_random_parameters('Dropout')
    network.add_layer_with_random_parameters('Dense')
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Conv2D')
    assert(network.get_network_layer_type(1) == 'Dropout')
    assert(network.get_network_layer_type(2) == 'Flatten')
    assert(network.get_network_layer_type(3) == 'Dense')
    
    
    print('network_is_1d_at_layer(layer_index) should correctly report if a network is 1d at a particular layer')
    network = Network()
    network.add_layer_with_random_parameters('Dense')    
    network.add_layer_with_random_parameters('Dense')    
    network.add_layer_with_random_parameters('Conv2D')
    network.check_network_structure()    

    assert(network.network_is_1d_at_layer(0) is True)
    assert(network.network_is_1d_at_layer(1) is True)
    assert(network.network_is_1d_at_layer(2) is False)
    assert(network.network_is_1d_at_layer(3) is False)

    print('network_is_2d_at_layer(layer_index) should correctly report if a network is 2d at a particular layer')
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.add_layer_with_random_parameters('Conv2D')
    network.add_layer_with_random_parameters('Dense')
    network.check_network_structure()    
    
    assert(network.network_is_2d_at_layer(0) is False)
    assert(network.network_is_2d_at_layer(1) is True)
    assert(network.network_is_2d_at_layer(2) is True)
    assert(network.network_is_2d_at_layer(3) is False)
    assert(network.network_is_2d_at_layer(4) is False)
    
    
print('Running tests....')    
test_network()
print('...tests complete')

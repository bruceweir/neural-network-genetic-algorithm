# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:38:41 2017

@author: brucew
"""

from network import Network
from optimizer import Optimizer
from train import compile_model, get_closest_valid_reshape_for_given_scale


def test_network():
    
    print('network.add_layer_with_random_parameters("layer_type") should add a network layer of the requested layer_type.')
    
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
    
    print("\t3. A Reshape layer should be added between a 2D layer and a 1D layer.")
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.add_layer_with_random_parameters('Conv2D');
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Dense')
    assert(network.get_network_layer_type(1) == 'Reshape')
    assert(network.get_network_layer_type(2) == 'Conv2D')
    
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.add_layer_with_random_parameters('MaxPooling2D');
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Dense')
    assert(network.get_network_layer_type(1) == 'Reshape')
    assert(network.get_network_layer_type(2) == 'MaxPooling2D')
    
    print("\t4. A Reshape cannot be called between 2 2D layers")
    network = Network()
    network.add_layer_with_random_parameters('Conv2D')
    network.add_layer_with_random_parameters('Reshape')
    network.add_layer_with_random_parameters('Conv2D')
    network.check_network_structure()
    
    assert(network.get_network_layer_type(0) == 'Conv2D')
    assert(network.get_network_layer_type(1) == 'Conv2D')
    
    print("\t5. The first layer of a network cannot be a Dropout layer")
    network = Network()
    network.add_layer_with_random_parameters('Dropout')
    network.add_layer_with_random_parameters('Dense')
    network.check_network_structure()
    
    assert(len(network.network_layers) == 1)
    assert(network.get_network_layer_type(0) == 'Dense')
    
    print("\t6. The first layer of a network cannot be a Reshape layer (this gets inserted later if require during model compilation).")
    network = Network()
    network.add_layer_with_random_parameters('Reshape')
    network.add_layer_with_random_parameters('Conv2D')
    network.check_network_structure()
    
    assert(len(network.network_layers) == 1)
    assert(network.get_network_layer_type(0) == 'Conv2D')
    
    print('\t7. Dropout layers should be handled correctly, depending if they are following a 1d or 2d layer.')
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
    
    print('\t8. Two Reshape layers are not permitted to be next to each other.')    
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.add_layer_with_random_parameters('Reshape')
    network.add_layer_with_random_parameters('Reshape')
    network.add_layer_with_random_parameters('Conv2D')
    network.check_network_structure()
    
    assert(len(network.network_layers) == 3)
    assert(network.get_network_layer_type(0) == 'Dense')
    assert(network.get_network_layer_type(1) == 'Reshape')
    assert(network.get_network_layer_type(2) == 'Conv2D')
    
    print('\t9. The first layer in the network cannot be a Flatten() layer (if required, these will be added automatically at model compilation).')
    network = Network()
    network.add_layer_with_random_parameters('Conv2D')
    network.add_layer_with_random_parameters('Dense')
    network.check_network_structure()
    del network.network_layers[0]
    network.check_network_structure()
    assert(network.get_network_layer_type(0) != 'Flatten')
    
    print('\t10. A Flatten layer cannot directly follow a 1d layer')
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.network_layers.append({'layer_parameters': {}, 'layer_type': 'Flatten'})
    network.check_network_structure()
    assert(network.number_of_layers() == 1)
    assert(network.get_network_layer_type(0) == 'Dense')
    
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

    print('network_is_1d_at_layer(layer_index) should correctly determine if Dropout layers are 1d')
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.add_layer_with_random_parameters('Dropout')
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
    
    
    print('network_is_2d_at_layer(layer_index) should correctly determine if Dropout layers are 2d')
    network = Network()
    network.add_layer_with_random_parameters('Conv2D')
    network.add_layer_with_random_parameters('Dropout')
    network.add_layer_with_random_parameters('Dense')
    network.check_network_structure()
    
    assert(network.network_is_2d_at_layer(0) is True)
    assert(network.network_is_2d_at_layer(1) is True)
    assert(network.network_is_2d_at_layer(2) is False)
    assert(network.network_is_2d_at_layer(3) is False)
    
    print('Network.starts_with_2d_layer() reports if the first layer of network is 2d')
    network = Network()
    network.add_layer_with_random_parameters('Dense')
    network.check_network_structure()
    
    assert(network.starts_with_2d_layer() is False)
    
    network = Network()
    network.add_layer_with_random_parameters('Conv2D')
    network.check_network_structure()
    
    assert(network.starts_with_2d_layer() is True)
    
    print('Network.create_random_network(number_of_layers=3, auto_check = False) creates a random network with number_of_layers layers if auto_check is False')
    print('The network created by create_random_network() is not guaranteed to be compilable unless auto_check is True')
    network = Network()
    network.create_random_network()
    assert(len(network.network_layers) == 3)
    
    network = Network()
    network.create_random_network(20, True)
    print('Compiling auto_checked network...')
    compile_model(network, 10, (784,), (28, 28,1))
    print('...done compiling.')
    
    print('The network created by Network.create_random_network() needs to call check_network_structure() before it can be safely compiled')
    for i in range(10):
        network = Network()
        network.create_random_network(10)
        network.check_network_structure()
        print('Compiling checked, random model %d...' % i)
        compile_model(network, 10, (784, ), (28, 28, 1))
        print('...done compiling')
    
    print('Any network created with a forbidden_layer_types argument should not contain any layers of the types listed in the forbidden_layer_types array')
    for i in range(10):
        network = Network(['Conv2D'])
        network.create_random_network(10, True)
        for l in range(network.number_of_layers()):
            assert(network.get_network_layer_type(l) is not 'Conv2D')
    
    
def test_optimizer():
    
    print('optimizer.mutate(network) returns a network object that has either had a layer added, removed or altered. The returned network should compile')
    network = Network()
    network.create_random_network(3, True)
    
    args={'mutate_chance':0.2, 'random_select':0.1, 'retain':0.4, 'forbidden_layer_types':[], 'population':10, 'initial_network_length':1}
    optimizer = Optimizer(**args)
    for i in range(10):
        print('Testing compilation of mutated network: %d' % i)
        optimizer.mutate(network)
        model = compile_model(network, 10, (784, ), (28, 28, 1))
        del model
        print('...done compiling')

    print('optimizer.breed(mother, father) returns an array containing 2 children, randomly bred from the network_layers of the parents. The two children should compile')
    father = Network()
    father.create_random_network(3, True)
    mother = Network()
    mother.create_random_network(3, True)
    print('Original father\n%s' % father.network_layers)        
    print('Original mother\n%s' % mother.network_layers)
    children = optimizer.breed(mother, father)
    print('Compiling children of first generation...')
    compile_model(children[0], 10, (784, ), (28, 28, 1))
    compile_model(children[1], 10, (784, ), (28, 28, 1))
    print('...compilation done')
    print('Testing 10 breeding generations')
    for i in range(10):
        print('Test %d' % i)
        mother.create_random_network(10, True)
        father.create_random_network(10, True)
        children = optimizer.breed(mother, father)
        print('Compiling children of generation %d...' % i)
        compile_model(children[0], 10, (784, ), (28, 28, 1))
        compile_model(children[1], 10, (784, ), (28, 28, 1))
        print('...compilation done')
    
    print ('optimizer.create_population(count, initial_length) creates and returns an array containing count networks or length initial_length (unless the network checker adds layers)')
    pop = optimizer.create_population(10)
    assert(len(pop) == 10)
    
    print('optimizer.evolve(pop) takes a population of networks and breeds and mutates them')
    pop = optimizer.create_population(10)
    optimizer.evolve(pop)
    
    #pop = [network1, network2]
    
def to_do():
    print('TODO')
    print('\t1. Move to Functional keras API')
    
print('Running tests....')    

test_network()
test_optimizer()
#test_train()
print('...tests complete')

to_do()
    
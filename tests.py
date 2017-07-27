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
    
    layer_id = network.add_layer_with_random_parameters('Dense')
    
    assert(network.get_network_layer_type(layer_id) == 'Dense')
   
    
    print('Network.create_random_network(number_of_layers=3) creates a random network with number_of_layers layers if auto_check is False')
    network = Network()
    network.create_random_network()
    assert(network.number_of_layers() == 3)
    
    for i in range(10):
        network = Network()
        network.create_random_network(20)
        print('Compiling auto_checked network...%d'% i)
        compile_model(network, 10, (784,), (28, 28,1))
        print('...done compiling.')
        
    
    print('Any network created with a forbidden_layer_types argument should not contain any layers of the types listed in the forbidden_layer_types array')
    for i in range(10):
        network = Network(['Conv2D'])
        network.create_random_network(10)
        for l in range(network.number_of_layers()):
            assert(network.get_network_layer_type(l) is not 'Conv2D')
    
def test_network_graph():

    print('Testing the network graph implementation.')
    print('\t. The network is directional, but it should be possible to trace the upstream and downstream connections for each layer.')
    network = Network()
    first_node_id = network.add_random_layer()
    assert(network.network_graph.number_of_nodes() == 1)
    
    second_node_id = network.add_random_layer(True, [first_node_id])
    assert(network.network_graph.number_of_nodes() == 2)
    
    assert(len(network.get_downstream_layers(first_node_id)) == 1)
    assert(len(network.get_downstream_layers(second_node_id)) == 0)
    
    assert(network.get_downstream_layers(first_node_id)[0] == second_node_id)
    
    assert(len(network.get_upstream_layers(second_node_id)) == 1)
    assert(network.get_upstream_layers(second_node_id)[0] == first_node_id)
    
    print('\t Deleting a layer should correctly rearrange the upstream and downstream edges.')
    network = Network()
    first_node_id = network.add_random_layer()
    second_node_id = network.add_random_layer(True, [first_node_id])
    third_node_id = network.add_random_layer(True, [second_node_id])
    
    network.delete_layer(second_node_id)
    assert(len(network.get_downstream_layers(first_node_id)) == 1)
    assert(network.get_downstream_layers(first_node_id)[0] == third_node_id)
    
    assert(len(network.get_upstream_layers(third_node_id)) == 1)
    assert(network.get_upstream_layers(third_node_id)[0] == first_node_id)
    
    assert(network.network_graph.has_node(second_node_id) == False)
    
    print('\t When deleting a layer, all the upstream layers should end up connected to all the downstream layers.')
    network = Network()
    first_level_node_1 = network.add_random_layer()
    first_level_node_2 = network.add_random_layer()
    second_level_node_1 = network.add_random_layer()
    
    network.connect_layers([first_level_node_1], [second_level_node_1])
    network.connect_layers([first_level_node_2], [second_level_node_1])
    
    third_level_node_1 = network.add_random_layer()
    third_level_node_2 = network.add_random_layer()
    third_level_node_3 = network.add_random_layer()
    
    network.connect_layers([second_level_node_1], [third_level_node_1])
    network.connect_layers([second_level_node_1], [third_level_node_2])
    network.connect_layers([second_level_node_1], [third_level_node_3])
    
    assert(len(network.get_upstream_layers(second_level_node_1)) == 2)
    assert(network.get_upstream_layers(second_level_node_1)[0] == first_level_node_1)
    assert(network.get_upstream_layers(second_level_node_1)[1] == first_level_node_2)
    
    assert(len(network.get_downstream_layers(second_level_node_1)) == 3)
    assert(network.get_downstream_layers(second_level_node_1)[0] == third_level_node_1)
    assert(network.get_downstream_layers(second_level_node_1)[1] == third_level_node_2)
    assert(network.get_downstream_layers(second_level_node_1)[2] == third_level_node_3)
    
    network.delete_layer(second_level_node_1)
    assert(network.network_graph.has_node(second_level_node_1) is False)
    
    assert(len(network.get_downstream_layers(first_level_node_1)) == 3)
    assert(network.get_downstream_layers(first_level_node_1)[0] == third_level_node_1)
    assert(network.get_downstream_layers(first_level_node_1)[1] == third_level_node_2)
    assert(network.get_downstream_layers(first_level_node_1)[2] == third_level_node_3)
    
    assert(len(network.get_downstream_layers(first_level_node_2)) == 3)
    assert(network.get_downstream_layers(first_level_node_2)[0] == third_level_node_1)
    assert(network.get_downstream_layers(first_level_node_2)[1] == third_level_node_2)
    assert(network.get_downstream_layers(first_level_node_2)[2] == third_level_node_3)
    
    assert(len(network.get_upstream_layers(third_level_node_1)) == 2)
    assert(network.get_upstream_layers(third_level_node_1)[0] == first_level_node_1)
    assert(network.get_upstream_layers(third_level_node_1)[1] == first_level_node_2)
    
    assert(len(network.get_upstream_layers(third_level_node_2)) == 2)
    assert(network.get_upstream_layers(third_level_node_2)[0] == first_level_node_1)
    assert(network.get_upstream_layers(third_level_node_2)[1] == first_level_node_2)
    
    assert(len(network.get_upstream_layers(third_level_node_3)) == 2)
    assert(network.get_upstream_layers(third_level_node_3)[0] == first_level_node_1)
    assert(network.get_upstream_layers(third_level_node_3)[1] == first_level_node_2)

    print('Inserting a layer should result in the connection between the upstream and downstream layer being rerouted through the new layer.')
    network = Network()
    first_layer_id = network.add_random_layer()
    second_layer_id = network.add_random_layer(True, [first_layer_id])
    
    inserted_layer_id = network.insert_random_layer(True, [first_layer_id], [second_layer_id])
    
    assert(len(network.get_downstream_layers(first_layer_id)) == 1)
    assert(network.get_downstream_layers(first_layer_id)[0] == inserted_layer_id)
    assert(len(network.get_downstream_layers(inserted_layer_id)) == 1)
    assert(network.get_downstream_layers(inserted_layer_id)[0] == second_layer_id)
    
    print('\t The layer type and parameters should be retrievable using network.get_network_layer_details(layer_id).')
    network = Network()
    layer_id = network.add_layer_with_random_parameters('Dropout')
    
    layer_type, layer_parameters = network.get_network_layer_details(layer_id)
    assert(layer_type == 'Dropout')
    assert('remove_probability' in layer_parameters)
    
    print('\t The layer type alone should be retrievable using network.get_network_layer_type(layer_id).')
    layer_type = network.get_network_layer_type(layer_id)
    assert(layer_type == 'Dropout')
    
    print('\t The layer parameters alone should be retrievable using network.get_network_layer_parameters(layer_id).')
    layer_parameters = network.get_network_layer_parameters(layer_id)
    assert('remove_probability' in layer_parameters)
    
    print('\t network.change_network_layer_parameter(layer_id, parameter, value) should change the parameter value of a layer.')
    layer_parameters = network.get_network_layer_parameters(layer_id)
    assert(layer_parameters['remove_probability'] != 99)
    
    network.change_network_layer_parameter(layer_id, 'remove_probability', 99)
    layer_parameters = network.get_network_layer_parameters(layer_id)
    assert(layer_parameters['remove_probability'] == 99)
    
    print('\t network.number_of_layers() should return the number of layers in the network.')
    network = Network()
    layer_id = network.add_random_layer()
    assert(network.number_of_layers() == 1)
    
    network.add_random_layer(True, [layer_id])
    assert(network.number_of_layers() == 2)
    
    
    
    
    
    

def test_optimizer():
    
    print('optimizer.mutate(network) returns a network object that has either had a layer added, removed or altered. The returned network should compile')
    network = Network()
    network.create_random_network(3)
    
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
    father.create_random_network(3)
    mother = Network()
    mother.create_random_network(3)
    print('Original father\n%s' % father.print_network_details())        
    print('Original mother\n%s' % mother.print_network_details())
    children = optimizer.breed(mother, father)
    print('Compiling children of first generation...')
    compile_model(children[0], 10, (784, ), (28, 28, 1))
    compile_model(children[1], 10, (784, ), (28, 28, 1))
    print('...compilation done')
    print('Testing 10 breeding generations')
    for i in range(10):
        print('Test %d' % i)
        mother.create_random_network(10)
        father.create_random_network(10)
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
    print('\t1. Add branching network structures')
    print('\t1\tSort out concatenation of different sized tensors')
    print('\t2. Add OOM capture and recovery during training')
    print('\t3. Add support for multiple input/output layers')    
print('Running tests....')    

test_network()
test_network_graph()
test_optimizer()
#test_train()
print('...tests complete')

to_do()
    
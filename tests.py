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
    
    for i in range(10):
        network = Network()
        network.create_random_network(20, True)
        print('Compiling auto_checked network...%d'% i)
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
    
def test_network_graph():

    print('Testing the network graph implementation')
    print('\t. The network is directional, but it should be possible to trace the upstream and downstream connections for each layer')
    network = Network()
    first_node_id = network.add_random_layer()
    assert(network.network_graph.number_of_nodes() == 1)
    
    second_node_id = network.add_random_layer(True, first_node_id)
    assert(network.network_graph.number_of_nodes() == 2)
    
    assert(len(network.get_downstream_layers(first_node_id)) == 1)
    assert(len(network.get_downstream_layers(second_node_id)) == 0)
    
    assert(network.get_downstream_layers(first_node_id)[0] == second_node_id)
    
    assert(len(network.get_upstream_layers(second_node_id)) == 1)
    assert(network.get_upstream_layers(second_node_id)[0] == first_node_id)
    
    print('\t Deleting a layer should correctly rearrange the upstream and downstream edges')
    network = Network()
    first_node_id = network.add_random_layer()
    second_node_id = network.add_random_layer(True, first_node_id)
    third_node_id = network.add_random_layer(True, second_node_id)
    
    network.delete_layer(second_node_id)
    assert(len(network.get_downstream_layers(first_node_id)) == 1)
    assert(network.get_downstream_layers(first_node_id)[0] == third_node_id)
    
    assert(len(network.get_upstream_layers(third_node_id)) == 1)
    assert(network.get_upstream_layers(third_node_id)[0] == first_node_id)
    
    assert(network.network_graph.has_node(second_node_id) == False)
    
    print('\t When deleting a layer, all the upstream layers should end up connected to all the downstream layers')
    network = Network()
    first_level_node_1 = network.add_random_layer()
    first_level_node_2 = network.add_random_layer()
    second_level_node_1 = network.add_random_layer()
    
    network.connect_layers(first_level_node_1, second_level_node_1)
    network.connect_layers(first_level_node_2, second_level_node_1)
    
    third_level_node_1 = network.add_random_layer()
    third_level_node_2 = network.add_random_layer()
    third_level_node_3 = network.add_random_layer()
    
    network.connect_layers(second_level_node_1, third_level_node_1)
    network.connect_layers(second_level_node_1, third_level_node_2)
    network.connect_layers(second_level_node_1, third_level_node_3)
    
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

    print('Inserting a layer should result in the connection between the upstream and downstream layer being rerouted through the new layer')
    network = Network()
    first_layer_id = network.add_random_layer()
    second_layer_id = network.add_random_layer(True, first_layer_id)
    
    inserted_layer_id = network.insert_random_layer(True, first_layer_id, second_layer_id)
    
    assert(len(network.get_downstream_layers(first_layer_id)) == 1)
    assert(network.get_downstream_layers(first_layer_id)[0] == inserted_layer_id)
    assert(len(network.get_downstream_layers(inserted_layer_id)) == 1)
    assert(network.get_downstream_layers(inserted_layer_id)[0] == second_layer_id)
    
    print('\t The layer type and parameters should be retrievable using network.get_network_layer_details(layer_id)')
    network = Network()
    layer_id = network.add_layer_with_random_parameters('Dropout')
    
    layer_type, layer_parameters = network.get_network_layer_details(layer_id)
    assert(layer_type == 'Dropout')
    assert('remove_probability' in layer_parameters)
    
    print('\t The layer type alone should be retrievable using network.get_network_layer_type(layer_id)')
    layer_type = network.get_network_layer_type(layer_id)
    assert(layer_type == 'Dropout')
    
    print('\t The layer parameters alone should be retrievable using network.get_network_layer_parameters(layer_id)')
    layer_parameters = network.get_network_layer_parameters(layer_id)
    assert('remove_probability' in layer_parameters)
    
    print('\t network.change_network_layer_parameter(layer_id, parameter, value) should change the parameter value of a layer')
    layer_parameters = network.get_network_layer_parameters(layer_id)
    assert(layer_parameters['remove_probability'] != 99)
    
    network.change_network_layer_parameter(layer_id, 'remove_probability', 99)
    layer_parameters = network.get_network_layer_parameters(layer_id)
    assert(layer_parameters['remove_probability'] == 99)
    
    print('\t network.number_of_layers() should return the number of layers in the network')
    network = Network()
    layer_id = network.add_random_layer()
    assert(network.number_of_layers() == 1)
    
    network.add_random_layer(True, layer_id)
    assert(network.number_of_layers() == 2)
    #assert(False)

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
    print('\t1. Add branching network structures')
    
print('Running tests....')    

#test_network()
test_network_graph()
#test_optimizer()
#test_train()
print('...tests complete')

to_do()
    
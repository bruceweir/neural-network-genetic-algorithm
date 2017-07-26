"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score
from network_layer_options import *
#from keras.models import load_model
from keras.utils import plot_model
import networkx as nx
#from keras.layers import Dense, Dropout, Conv2D, Flatten, Reshape, MaxPooling2D

class Network():
    """Represent a network and let us operate on it.
    """

    def __init__(self, forbidden_layer_types=[]):
        """Initialize our network.

        """
        self.accuracy = 0.0
        self.nn_network_layer_options = self.create_network_layer_options()
        
        self.trained_model = None
        self.forbidden_layer_types = forbidden_layer_types
        self.network_graph = nx.DiGraph()
        self.layer_id = -1

    def get_new_layer_id(self):
        
        self.layer_id += 1
        return self.layer_id
    
    
    def create_random_network(self, number_of_layers=3):
        """Create a random network."""

        self.network_graph = nx.DiGraph()
        
        previous_layer_id = None

        for i in range(number_of_layers):
            allow_dropout = True
            if i == 0:
                allow_dropout = False

            previous_layer_id = self.add_random_layer(allow_dropout, [previous_layer_id])


    def add_random_layer(self, allow_dropout = True, upstream_layer_ids = None):

        return self.add_layer_with_parameters(self.create_random_layer(allow_dropout), upstream_layer_ids)


    def add_layer_with_random_parameters(self, layer_type, upstream_layer_ids = None):

        return self.add_layer_with_parameters(self.create_layer(layer_type), upstream_layer_ids)

    
    def add_layer_with_parameters(self, parameters, upstream_layer_ids):
        
        new_layer_id = self.get_new_layer_id()
        
        self.network_graph.add_node(new_layer_id, parameters)
        
        self.connect_layers(upstream_layer_ids, [new_layer_id])
        
        self.clear_trained_model()
        
        return new_layer_id


    def connect_layers(self, upstream_layer_ids, layer_ids):

        """ Add an edge between the layers whose ids are listed in the upstream_layer_ids array and each of the layers in the layer_ids list """        
    
        if upstream_layer_ids == None or layer_ids == None:
            return
                
        for upstream_layer_id in upstream_layer_ids:
            for layer_id in layer_ids:
                if self.network_graph.has_node(upstream_layer_id) and self.network_graph.has_node(layer_id):
                    self.network_graph.add_edge(upstream_layer_id, layer_id)


    def disconnect_layers(self, upstream_layer_ids, layer_ids):

        """ Remove edges between the layers whose ids are listed in the upstream_layer_ids array and each of the layers in the layer_ids list """        
    
        for upstream_layer_id in upstream_layer_ids:
            for layer_id in layer_ids:
                if self.network_graph.has_node(upstream_layer_id) and self.network_graph.has_node(layer_id):
                    self.network_graph.remove_edge(upstream_layer_id, layer_id)

    def insert_layer_with_random_parameters(self, layer_type, upstream_layer_ids, downstream_layer_ids):

        return self.insert_layer_between_layers(self.create_layer(layer_type), upstream_layer_ids, downstream_layer_ids)


    def insert_random_layer(self, allow_dropout, upstream_layer_ids, downstream_layer_ids):

        return self.insert_layer_between_layers(self.create_random_layer(allow_dropout), upstream_layer_ids, downstream_layer_ids)

    
    def insert_layer_with_parameters(self, parameters, upstream_layer_ids, downstream_layer_ids):
        
        return self.insert_layer_between_layers(parameters, upstream_layer_ids, downstream_layer_ids)
    

    def insert_layer_between_layers(self, layer, upstream_layer_ids, downstream_layer_ids):
        
        new_layer_id = self.get_new_layer_id()
        self.network_graph.add_node(new_layer_id, layer)
        
        self.disconnect_layers(upstream_layer_ids, downstream_layer_ids)
            
        self.connect_layers(upstream_layer_ids, [new_layer_id]);
        self.connect_layers([new_layer_id], downstream_layer_ids)
            
        self.clear_trained_model()
        
        return new_layer_id
                

    def delete_layer(self, layer_id):
        
        if self.network_graph.has_node(layer_id) is not True:
            return
        
        upstream_layers = self.get_upstream_layers(layer_id)
        downstream_layers = self.get_downstream_layers(layer_id)
        
        self.disconnect_layers(upstream_layers, [layer_id])
        self.disconnect_layers([layer_id], downstream_layers)
        
        self.network_graph.remove_node(layer_id)
        
        self.connect_layers(upstream_layers, downstream_layers)
                
        self.clear_trained_model()

    def get_upstream_layers(self, layer_id):
        
        if self.network_graph.has_node(layer_id) is not True:        
            return []
        
        return self.network_graph.reverse().neighbors(layer_id)

    
    def get_downstream_layers(self, layer_id):
        
        if self.network_graph.has_node(layer_id) is not True:
            return []
        
        return self.network_graph.neighbors(layer_id)
    
    
       
    def change_network_layer_parameter(self, layer_id, parameter, value):

        parameters = self.get_network_layer_parameters(layer_id)

        if parameter in parameters:
            parameters[parameter] = value
            print('Network.change_network_layer_parameter() %s, %s' % (parameter, value))
            self.clear_trained_model()
        else:
            raise ValueError('Network.change_network_layer_parameter(). Unknown parameter')

        self.clear_trained_model()


    def create_random_layer(self, allow_dropout=False):

        # [:] creates a new list, rather than copying by reference
        layers_not_to_select = self.forbidden_layer_types[:]

        if allow_dropout == False:
            layers_not_to_select.append('Dropout')

        layer_type = random.choice([choice for choice in self.nn_network_layer_options['LayerTypes'] if choice not in layers_not_to_select])

        return self.create_layer(layer_type)


    def create_layer(self, layer_type):

        layer_parameters = {}

        if layer_type != 'Flatten':
            for key in self.nn_network_layer_options[layer_type]:
                layer_parameters[key] = random.choice(self.nn_network_layer_options[layer_type][key])

        return {'layer_type': layer_type, 'layer_parameters': layer_parameters}
    
    
    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.trained_model == None:
            self.accuracy = train_and_score(self, dataset)

    def log_network(self):
        """Print out a network."""
        logging.info(self.network_layers)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
        
    def create_network_layer_options(self):
               
        nn_network_layer_options = {
                'LayerTypes': get_layer_types_for_random_selection(),
                'Dense': get_dense_layer_options(),
                'Conv2D': get_conv2d_layer_options(),
                'Dropout': get_dropout_layer_options(),  
                'Reshape': get_reshape_layer_options(),
                'MaxPooling2D': get_maxpooling2d_layer_options()
        }
        
        return nn_network_layer_options

    
    def change_random_parameter_for_layer(self, index):
        
        layer_type = self.get_network_layer_type(index)
        parameter = self.get_random_parameter_for_layer_type(layer_type)
        current_value = self.get_value_of_parameter_for_layer(index, parameter)
        new_value = self.get_random_value_of_parameter_for_layer_type(layer_type, parameter, current_value)                
        self.change_network_layer_parameter(index, parameter, new_value)
        
        
    def get_random_parameter_for_layer_type(self, layer_type):
        
        option_function = self.get_option_function_for_layer_type(layer_type)
        
        parameter = random.choice(list(option_function().keys()))
          
        return parameter


    def get_random_value_of_parameter_for_layer_type(self, layer_type, parameter, value_to_exclude=None):
        
        option_function = self.get_option_function_for_layer_type(layer_type)
        value = random.choice([choice for choice in option_function()[parameter] if choice != value_to_exclude])

        return value        

            
    def get_option_function_for_layer_type(self, layer_type):
        
        option_function = None
        
        if layer_type == 'Dense':
            option_function = get_dense_layer_options
            
        elif layer_type == 'Conv2D':
            option_function = get_conv2d_layer_options
            
        elif layer_type == 'Dropout':
            option_function = get_dropout_layer_options
            
        elif layer_type == 'Reshape':
            option_function = get_reshape_layer_options
            
        elif layer_type == 'MaxPooling2D':
            option_function = get_maxpooling2d_layer_options
            
        else:
            raise NameError('Error: unknown layer_type: %s' % layer_type)
        
        return option_function

    def get_value_of_parameter_for_layer(self, index, parameter):
                 
        return self.get_network_layer_parameters(index)[parameter]


    def print_network_details(self):
        print(self.network_graph.node.items())
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
        
        
    def get_network_layer_type(self, layer_id):
        
        layer_type, _ = self.get_network_layer_details(layer_id)
        
        return layer_type
    
    
    def get_network_layer_parameters(self, layer_id):
        
        _, parameters = self.get_network_layer_details(layer_id)
        
        return parameters
    
    
    def get_network_layer_details(self, layer_id):
        
        layer_info = self.get_network_layer_details_dictionary(layer_id)
        
        if layer_info == None:
            return None, None
        
        return layer_info['layer_type'], layer_info['layer_parameters']
 
    def get_network_layer_details_dictionary(self, layer_id):
        
        layer_info = [layer_info for node_id, layer_info in self.network_graph.node.items() if node_id == layer_id]
        
        if len(layer_info) == 0:
            return None
        
        return layer_info[0]
    
        
    def get_all_network_layer_ids(self):
        
        layer_ids = [node_id for node_id, layer_info in self.network_graph.node.items()]
        return layer_ids
    
    def get_network_layers_with_no_downstream_connections(self):
        
        return [x for x in self.get_all_network_layer_ids() if len(self.get_downstream_layers(x)) == 0]
        
    def number_of_layers(self):
        return len(self.network_graph)
    
    def has_a_layer_with_id(self, layer_id):
        
        return self.network_graph.has_node(layer_id)
    
    
    def save_network_details(self, file_name_prepend):
        
        self.save_model(file_name_prepend + ".h5")
        self.save_model_image(file_name_prepend + ".png")

        
    def save_model(self, fileName):
        if self.trained_model is not None and len(fileName) is not 0:
            print('Saving model to %s' % fileName)
            logging.info('Saving model to %s' % fileName)            
            self.trained_model.save(fileName)
            
    def save_model_image(self, fileName):
        if self.trained_model is not None and len(fileName) is not 0:
            plot_model(self.trained_model, to_file=fileName, show_shapes=True)
    
        
    def clear_trained_model(self):
        if self.trained_model is not None:
            del self.trained_model
            self.trained_model = None

    
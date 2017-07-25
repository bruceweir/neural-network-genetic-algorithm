"""Class that represents the network to be evolved."""
import random
import logging
import json
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
    
    
    def create_random_network(self, number_of_layers=3, auto_check=False):
        """Create a random network."""

        self.network_graph = nx.DiGraph()

        for i in range(number_of_layers):
            allow_dropout = True
            if i == 0:
                allow_dropout = False

            self.add_random_layer(allow_dropout)


        if auto_check is True:
            self.check_network_structure()

    def add_random_layer(self, allow_dropout = True, upstream_layer_id = None):

        
        new_layer_id = self.get_new_layer_id()
        self.network_graph.add_node(new_layer_id, self.create_random_layer(allow_dropout))
        
        if upstream_layer_id is not None:
            self.connect_layers(upstream_layer_id, new_layer_id)
        
        self.clear_trained_model()
        
        return new_layer_id

    def connect_layers(self, upstream_layer_id, layer_id):
        
        if self.network_graph.has_node(upstream_layer_id) is not True or self.network_graph.has_node(layer_id) is not True:
            return
        
        self.network_graph.add_edge(upstream_layer_id, layer_id)


    def add_layer_with_random_parameters(self, layer_type, upstream_layer_id = None):

        new_layer_id = self.get_new_layer_id()       
        self.network_graph.add_node(new_layer_id, self.create_layer(layer_type))
        
        if upstream_layer_id is not None:
            self.connect_layers(upstream_layer_id, new_layer_id)
            
        
        self.clear_trained_model()
        
        return new_layer_id

    def insert_layer_with_random_parameters(self, layer_type, upstream_layer_id = None, downstream_node_id= None):

        return self.insert_layer_between_layers(self.create_layer(layer_type), upstream_layer_id, downstream_node_id)


    def insert_random_layer(self, allow_dropout=True, upstream_layer_id = None, downstream_node_id= None):

        return self.insert_layer_between_layers(self.create_random_layer(allow_dropout), upstream_layer_id, downstream_node_id)


    def insert_layer_between_layers(self, layer, upstream_layer_id, downstream_layer_id):
        
        new_layer_id = self.get_new_layer_id()
        self.network_graph.add_node(new_layer_id, layer)
        
        if upstream_layer_id is not None and downstream_layer_id is not None:
            self.network_graph.remove_edge(upstream_layer_id, downstream_layer_id)           
            
        if upstream_layer_id is not None:
            self.connect_layers(upstream_layer_id, new_layer_id);
        if downstream_layer_id is not None:
            self.connect_layers(new_layer_id, downstream_layer_id)
            
        self.clear_trained_model()
        
        return new_layer_id
                

    def delete_layer(self, layer_id):
        
        if self.network_graph.has_node(layer_id) is not True:
            return
        
        upstream_layers = self.get_upstream_layers(layer_id)
        downstream_layers = self.get_downstream_layers(layer_id)
        
        self.network_graph.remove_node(layer_id)
        
        for up in upstream_layers:
            for down in downstream_layers:
                self.connect_layers(up, down)
                
        self.clear_trained_model()

    def get_upstream_layers(self, node_id):
        
        if self.network_graph.has_node(node_id) is not True:        
            return []
        
        return self.network_graph.reverse().neighbors(node_id)

    
    def get_downstream_layers(self, node_id):
        
        if self.network_graph.has_node(node_id) is not True:
            return []
        
        return self.network_graph.neighbors(node_id)
    
    
       
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


    def check_network_structure(self):

        """ Apply various rules for allowing only certain network structures

        Insert a Flatten() layer if going from a 2D layer to a Dense layer.
        Dropout layers cannot immediately follow Dropout layers.
        Insert a Reshape() layer when going from a 1d layer to a Conv2D layer.
        Do not allow 2D to 2D reshapes
        The first layer cannot be a Dropout layer
        """
        i = 1

        network_changed = False

        while i < len(self.network_layers):
            #print('%d: %s' % (i, self.network_layers[i]['layer_type']))
            current_layer_type = self.get_network_layer_type(i)
            previous_layer_type = self.get_network_layer_type(i-1)

            if current_layer_type == 'Dense' and self.network_is_not_1d_at_layer(i-1):
                self.network_layers.insert(i, {'layer_type':'Flatten', 'layer_parameters':{}})
                network_changed = True
                i = 1
            elif current_layer_type == 'Dropout' and previous_layer_type == 'Dropout':
                del self.network_layers[i]
                network_changed = True
                i = 1
            elif self.network_is_2d_at_layer(i) and current_layer_type != 'Reshape' and self.network_is_not_2d_at_layer(i-1):# (current_layer_type == 'Conv2D' or current_layer_type == 'MaxPooling2D') and self.network_is_not_2d_at_layer(i-1):
                self.insert_layer_with_random_parameters(i, 'Reshape')
                network_changed = True
                i = 1
            elif current_layer_type == 'Reshape' and self.network_is_2d_at_layer(i-1):
                del self.network_layers[i]
                network_changed = True
                i = 1
            elif current_layer_type == 'Flatten' and self.network_is_1d_at_layer(i-1):
                del self.network_layers[i]
                network_changed = True
                i = 1
                               
            else:
                i+=1

        forbidden_first_layers = ['Dropout', 'Reshape', 'Flatten']
                
        while  self.number_of_layers() > 0 and self.get_network_layer_type(0) in forbidden_first_layers:
            del self.network_layers[0]
            network_changed = True
            
        if network_changed is True:
            self.clear_trained_model()


    def network_is_not_1d_at_layer(self, layer_index):
        
        return not self.network_is_1d_at_layer(layer_index)
    
    
    def network_is_1d_at_layer(self, layer_id):
               
        
        while(True):
            if self.network_graph.has_node(layer_id) is False:
                raise ValueError('Network.network_is_1d_at_layer(). Unknown layer_id')
    
                    
            layer_type = self.get_network_layer_type(layer_id)
            
            if layer_type == 'Dense' or layer_type == 'Flatten':
                return True
            if '2D' in layer_type or layer_type == 'Reshape':
                return False
            
            upstream_layer_ids = self.get_upstream_layers(layer_id)
            
            return all([self.network_is_1d_at_layer(layer_id) for layer_id in upstream_layer_ids])
                
            
        return True

    
    def starts_with_2d_layer(self):
        if self.number_of_layers() == 0:
            return False
        
        return self.network_is_2d_at_layer(0)
    
    
    def network_is_not_2d_at_layer(self, layer_index):
        return not self.network_is_2d_at_layer(layer_index)
    
    
    def network_is_2d_at_layer(self, layer_id):
        
        
        if self.network_graph.has_node(layer_id) is False:
            raise ValueError('network.network_is_2d_at_layer(). Unknown layer_id')
            
        layer_type = self.get_network_layer_type(layer_id)
        
        if '2D' in layer_type or layer_type == 'Reshape':
            return True
        if layer_type == 'Dropout':
            upstream_layer_ids = self.get_upstream_layers(layer_id)        
            return all([self.network_is_2d_at_layer(layer_id) for layer_id in upstream_layer_ids])
            
        return False

    def layer_type_is_not_1d(self, layer_type):
        
        return not self.layer_type_is_1d(layer_type)
    
    
    def layer_type_is_1d(self, layer_type):
        
        return layer_type in ['Dense']
    
    
    def layer_type_is_not_2d(self, layer_type):
        
        return not self.layer_type_is_2d(layer_type)
    
    
    def layer_type_is_2d(self, layer_type):
        
        return '2D' in layer_type
    
    
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


    def print_network_as_json(self, just_the_layers=False):
        
        if just_the_layers is True:
            for layer in self.network_layers:
                print(layer['layer_type'])
        else:
            print(json.dumps(self.network_layers, indent=4))

    def print_network_details(self):
        self.print_network_as_json()
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
        
        
    def get_network_layer_type(self, layer_id):
        
        layer_type, _ = self.get_network_layer_details(layer_id)
        
        return layer_type
    
    
    def get_network_layer_parameters(self, layer_id):
        
        _, parameters = self.get_network_layer_details(layer_id)
        
        return parameters
    
    
    def get_network_layer_details(self, layer_id):
        
        layer_info = [(layer_info) for node_id, layer_info in self.network_graph.node.items() if node_id == layer_id]
        
        if len(layer_info) == 0:
            return None, None
        
        return layer_info[0]['layer_type'], layer_info[0]['layer_parameters']
 
    
    def number_of_layers(self):
        return len(self.network_graph)
    
    
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

    
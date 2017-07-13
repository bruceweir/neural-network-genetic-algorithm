"""Class that represents the network to be evolved."""
import random
import logging
import json
from train import train_and_score
from keras.models import load_model
from keras.utils import plot_model

class Network():
    """Represent a network and let us operate on it.
    """

    def __init__(self):
        """Initialize our network.

        """
        self.accuracy = 0.
        
        self.nn_network_layer_options = self.create_network_layer_options()
        self.network_layers = []  # (array): represents network parameters
        self.trained_model = None

    def create_random_network(self, number_of_layers=3, auto_check = False):
        """Create a random network."""
        
        self.network_layers = []
        
        for i in range(number_of_layers):
            allow_dropout = True
            if i==0:
                allow_dropout = False
            else:
                allow_dropout = True
                
            self.network_layers.append(self.create_random_layer(allow_dropout));
        
        if auto_check is True:
            self.check_network_structure()
            
           

    def add_layer_with_random_parameters(self, layer_type):
        
        self.network_layers.append(self.create_layer(layer_type));

    def insert_layer_with_random_parameters(self, layer_type, index):
        
        self.network_layers.insert(index, self.create_layer(layer_type))
            
    def create_random_layer(self, allow_dropout=False):
        
        if allow_dropout == True:
            layer_type = random.choice(self.nn_network_layer_options['LayerTypes'])
        else:
            layer_type = random.choice(self.nn_network_layer_options['LayerTypes'][:2])
    
        return self.create_layer(layer_type)
        
        
    def create_layer(self, layer_type):
        
        layer_parameters = {}
        
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
        i=1
        
        while i < len(self.network_layers):
            #print('%d: %s' % (i, self.network_layers[i]['layer_type']))
            current_layer_type = self.get_network_layer_type(i)           
            previous_layer_type = self.get_network_layer_type(i-1)
            
            if current_layer_type == 'Dense' and self.network_is_not_1d_at_layer(i-1):
                self.network_layers.insert(i, {'layer_type':'Flatten', 'layer_parameters':{}})
                i=1
            elif current_layer_type == 'Dropout' and previous_layer_type == 'Dropout':
                del self.network_layers[i]
                i=1
            elif current_layer_type == 'Conv2D' and self.network_is_not_2d_at_layer(i-1):
                self.insert_layer_with_random_parameters('Reshape', i)
                i=1
            elif current_layer_type == 'Reshape' and self.network_is_2d_at_layer(i-1):
                del self.network_layers[i]
                i=1
            elif current_layer_type == 'Flatten' and self.network_is_1d_at_layer(i-1):
                del self.network_layers[i]
                i=1
                               
            else:
                i+=1

        forbidden_first_layers = ['Dropout', 'Reshape', 'Flatten']
                
        while  self.number_of_layers() > 0 and self.get_network_layer_type(0) in forbidden_first_layers:
            del self.network_layers[0]


    def network_is_not_1d_at_layer(self, layer_index):
        
        return not self.network_is_1d_at_layer(layer_index)
    
    
    def network_is_1d_at_layer(self, layer_index):
               
        index = layer_index
        
        while index > -1:
            layer_type = self.network_layers[index]['layer_type']
            
            if layer_type == 'Dense' or layer_type == 'Flatten':
                return True
            if '2D' in layer_type or layer_type == 'Reshape':
                return False
            
            index = index-1
        
        return True

    
    def starts_with_2d_layer(self):
        return self.network_is_2d_at_layer(0)
    
    
    def network_is_not_2d_at_layer(self, layer_index):
        return not self.network_is_2d_at_layer(layer_index)
    
    
    def network_is_2d_at_layer(self, layer_index):
        layer = self.network_layers[layer_index]
        layer_type = layer['layer_type']
        
        if '2D' in layer_type or layer_type == 'Reshape':
            return True
        if layer_type == 'Dropout':
            return self.network_is_2d_at_layer(layer_index - 1)
        
        return False
    
    
    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network_layers = network

    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self, dataset)

    def log_network(self):
        """Print out a network."""
        logging.info(self.network_layers)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
        

    def create_network_layer_options(self):
        
        
        nn_network_layer_options = {
                'LayerTypes': self.get_layer_types_for_random_selection(),
                'Dense': self.get_dense_layer_options(),
                'Conv2D': self.get_conv2d_layer_options(),
                'Dropout': self.get_dropout_layer_options(),  
                'Reshape': self.get_reshape_layer_options()
        }
        
        return nn_network_layer_options

    def get_dense_layer_options(self):
    
        return {
                'nb_neurons': [64, 128, 256, 512, 768, 1024],
                'activation': ['relu', 'elu', 'tanh', 'sigmoid']           
        }
        
    def get_reshape_layer_options(self):
        
        return { 
                'first_dimension_scale': [1, 2, 4, 8, 16, 32] 
        }
    
    def get_conv2d_layer_options(self):
        
        return {
                'strides': [(1, 1), (2, 2), (4, 4)],
                'kernel_size': [(1, 1), (3, 3), (5, 5), (7, 7)],
                'nb_filters': [2, 8, 16, 32, 64],
                'activation': ['relu', 'elu', 'tanh', 'sigmoid']
        }
    
    def get_dropout_layer_options(self):
        
        return {
                'remove_probability':[.5, .3, .2]
        }
    
    
    def get_layer_types_for_random_selection(self):
        
        return ['Dense', 'Conv2D', 'Dropout']
    
    def print_network_as_json(self, just_the_layers=False):
        
        if just_the_layers is True:
            for layer in self.network_layers:
                print(layer['layer_type'])
        else:
            print(json.dumps(self.network_layers, indent=4))

    def print_network_details(self):
        self.print_network_as_json()
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
        
        
    def get_network_layer_type(self, index):
        return self.network_layers[index]['layer_type']
    
    def get_network_layer_parameters(self, index):
        return self.network_layers[index]['layer_parameters']
    
    def number_of_layers(self):
        return len(self.network_layers)
    
    def save_model(self, fileName):
        if self.trained_model is not None and len(fileName) is not 0:
            print('Saving model to %s', fileName)
            logging.info('Saving model to ', fileName)            
            self.trained_model.save(fileName)
            
    def save_model_image(self, fileName):
        plot_model(self.trained_model, to_file=fileName, show_shapes=True)
"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None, nn_network_layer_options=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.nn_network_layer_options = nn_network_layer_options
        self.network_layers = []  # (array): represents network parameters

    def create_random(self):
        """Create a random network."""
        
        self.network_layers = []
        
        for i in range(self.nn_network_layer_options['NbInitialNetworkLayers']):
            allow_dropout = True
            if i==0:
                allow_dropout = False
            else:
                allow_dropout = True
                
            self.network_layers.append(self.get_random_layer(allow_dropout));
            
            
    def get_random_layer(self, allow_dropout=False):
        
        if allow_dropout == True:
            layer = random.choice(self.nn_network_layer_options['LayerTypes'])
        else:
            layer = random.choice(self.nn_network_layer_options['LayerTypes'][:2])
            
        layer_parameters = {}
            
        if layer == 'Dense':
            for key in self.nn_network_layer_options['DenseOptions']:
                layer_parameters[key] = random.choice(self.nn_network_layer_options['DenseOptions'][key])
        if layer == 'Conv2D':
            for key in self.nn_network_layer_options['Conv2DOptions']:
                layer_parameters[key] = random.choice(self.nn_network_layer_options['Conv2DOptions'][key])
        if layer == 'Dropout':
            for key in self.nn_network_layer_options['DropoutOptions']:
                layer_parameters[key] = random.choice(self.nn_network_layer_options['DropoutOptions'][key])
        
        return {'layerType': layer, 'layer_parameters':layer_parameters}
    
    
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
            self.accuracy = train_and_score(self.network_layers, dataset)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network_layers)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))

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
        self.network = []  # (array): represents network parameters

    def create_random(self):
        """Create a random network."""
        for i in range(self.nn_network_layer_options['NbInitialHiddenLayers']):
            self.network.append(self.get_random_layer());
            
            
    def get_random_layer(self):
        
        layer = random.choice(self.nn_network_layer_options['LayerTypes'])
            
        layer_parameters = {}
            
        if layer == 'Dense':
            for key in self.nn_network_layer_options['DenseOptions']:
                layer_parameters[key] = random.choice(self.nn_network_layer_options['DenseOptions'][key])
        if layer == 'Convolution':
            for key in self.nn_network_layer_options['ConvolutionOptions']:
                layer_parameters[key] = random.choice(self.nn_network_layer_options['ConvolutionOptions'][key])
        if(layer == 'Dropout'):
            for key in self.nn_network_layer_options['DropoutOptions']:
                layer_parameters[key] = random.choice(self.nn_network_layer_options['DropoutOptions'][key])
        
        return {'layerType': layer, 'layer_parameters':layer_parameters}
    
    
    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network, dataset)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:41:05 2017

@author: brucew
"""

from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, Reshape, ActivityRegularization
from keras.layers import MaxPooling2D, Input, concatenate, ZeroPadding1D, ZeroPadding2D
import math
from functools import reduce


class Network_Compiler():
    
    """A class containing the functionality to create a Keras model from a 
    Network object """
    def __init__(self):
        self.list_of_trained_layers = []
        
    
    def compile_model(self, network, output_shape, input_shape, natural_input_shape, is_classification):
        """Compile a keras model.
    
        Args:
            network: A network object with a defined network structure
            nb_classes: The number of classification classes to output
            input_shape: Shape of the input vector (i.e. for training on MNIST: (784,))
            natural_input_shape: Shape of the input vector if it can be sensibly considered as a 2D image (i.e. for training on greyscale MNIST only: (28,28,1))
            
    
        Returns:
            a compiled Keras Model which can be trained.
    
        """
        
        self.list_of_trained_layers = [{'layer_id':x, 'compiled_layer':None} for x in network.get_all_network_layer_ids()]
        # Get our network parameters.
        
        inputs = Input(shape=input_shape)
        layer = inputs
    
        first_layer_ids = network.get_network_layers_with_no_upstream_connections()
        if len(first_layer_ids) > 1:
            # For now, connect the layers - TODO, deal with multiple inputs case
            network.connect_layers([first_layer_ids[0]], first_layer_ids[1:])
        
        last_layer_ids = network.get_network_layers_with_no_downstream_connections()
        if len(last_layer_ids) > 1:
            # For now, connect the layers - TODO, deal with multiple outputs case        
            network.connect_layers(last_layer_ids[1:], [last_layer_ids[0]])
        
        final_layer_id = last_layer_ids[0]
        
        layer = self.add_layer(network, final_layer_id, layer, natural_input_shape)
       
        _, _, number_of_dimensions_in_previous_layer = self.get_compiled_layer_shape_details(layer)
        
        
        if is_classification:
        
            if number_of_dimensions_in_previous_layer > 2:
                layer = Flatten()(layer)
        
            output = self.add_dense_layer({'activation': 'softmax', 'nb_neurons': output_shape[0]}, layer)
    
            model = Model(inputs=inputs, outputs=output, name='Output')
        
            model.compile(loss='categorical_crossentropy', optimizer='adam',
                          metrics=['accuracy'])
        else:
            if number_of_dimensions_in_previous_layer > 2:
                layer = Flatten()(layer)
        
            layer = self.add_dense_layer({'activation': 'linear', 'nb_neurons': self.calculate_number_of_neurons_in_shape(output_shape)}, layer)
            output = Reshape(output_shape)(layer)
            model = Model(inputs=inputs, outputs=output, name='Output')
        
            model.compile(loss='mse', optimizer='adam',
                          metrics=['accuracy'])
    
        return model


    def add_layer(self, network, layer_id, input_layer, natural_input_shape):
        
        """ Starting at the output layer, this should recurse up the network graph, adding Keras layers """
        
        if self.get_compiled_layer_for_id(layer_id) is not None:
            return self.get_compiled_layer_for_id(layer_id)
        
        
        layer = input_layer
        
        layers_input_into_this_level = []
        
        for ids in network.get_upstream_layers(layer_id):
            layers_input_into_this_level.append(self.add_layer(network, ids, input_layer, natural_input_shape))
    
        
        if len(layers_input_into_this_level) > 1:
            layer = self.add_concatenation(layers_input_into_this_level)       
        
        elif len(layers_input_into_this_level) == 1:        
            layer = layers_input_into_this_level[0]
    
        layer_type = network.get_network_layer_type(layer_id)
        layer_parameters = network.get_network_layer_parameters(layer_id)
    
        
     
        if layer_type == 'Dense':
            layer = self.add_dense_layer(layer_parameters, layer)
            
        elif layer_type == 'Conv2D':            
            layer = self.add_conv2D_layer(layer_parameters, layer, natural_input_shape)            
                
        elif layer_type == 'MaxPooling2D':
            layer = self.add_maxpooling2d_layer(layer_parameters, layer, natural_input_shape)
                
        elif layer_type == 'Dropout':
            layer = self.add_dropout_layer(layer_parameters, layer)
        
        elif layer_type == 'ActivityRegularization':
            layer = self.add_activity_regularization_layer(layer_parameters, layer)
        else:
            raise ValueError('add_layer(), unknown layer type: ' + layer_type)
        
        previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer = self.get_compiled_layer_shape_details(layer)
    
        self.set_compiled_layer_for_id(layer_id, layer)
        
        return layer
    
    
    def add_dense_layer(self, layer_parameters, input_layer):
        
        try:
            layer = Dense(layer_parameters['nb_neurons'], 
                          activation=layer_parameters['activation'])(input_layer)
        except:
            print('add_dense_layer: Flattening input')
            layer = Flatten()(input_layer)
            layer = Dense(layer_parameters['nb_neurons'], 
                      activation=layer_parameters['activation'])(layer)
        return layer

    
    def add_conv2D_layer(self, layer_parameters, input_layer, natural_input_shape):
        
        previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer = self.get_compiled_layer_shape_details(input_layer)
        
        aspect_ratio = self.get_aspect_ratio(natural_input_shape)
        
        try:
            kernel_size = self.get_checked_2d_kernel_size_for_layer(previous_layer_shape, layer_parameters['kernel_size'])
            layer = Conv2D(layer_parameters['nb_filters'], 
                             kernel_size=kernel_size, 
                             strides=layer_parameters['strides'], 
                             activation=layer_parameters['activation'])(input_layer)
        except (ValueError, IndexError):
            print('add_conv2D_layer: Reshaping input')
            reshape_size = self.get_reshape_size_closest_to_aspect_ratio(number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer, aspect_ratio)
            layer= Reshape(reshape_size)(input_layer)       
            layer = self.add_conv2D_layer(layer_parameters, layer, natural_input_shape);
            
        return layer

            
    def add_maxpooling2d_layer(self, layer_parameters, input_layer, natural_input_shape):
        
        previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer = self.get_compiled_layer_shape_details(input_layer)
        
        aspect_ratio = self.get_aspect_ratio(natural_input_shape)
        
        try:
                pool_size = self.get_checked_2d_kernel_size_for_layer(previous_layer_shape, layer_parameters['pool_size'])            
                layer = MaxPooling2D(pool_size=pool_size)(input_layer)
        
        except (ValueError, IndexError):
            reshape_size = self.get_reshape_size_closest_to_aspect_ratio(number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer, aspect_ratio)
            layer= Reshape(reshape_size)(input_layer)
            layer = self.add_maxpooling2d_layer(layer_parameters, layer, natural_input_shape)
            
        return layer


    def add_dropout_layer(self, layer_parameters, input_layer):
        
        layer = Dropout(layer_parameters['remove_probability'])(input_layer)
        return layer
       
    def add_activity_regularization_layer(self, layer_parameters, input_layer):
        
        l1 = layer_parameters.get('l1', 0.0)
        l2 = layer_parameters.get('l2', 0.0)
        layer= ActivityRegularization(l1, l2)(input_layer)
        return layer
    
    
    def add_concatenation(self, input_layers):
        
        try:
            layer = concatenate(input_layers)
        except ValueError:
            
            reshape_size = self.calculate_best_size_for_concatenated_layers(input_layers)
            
            for x in range(len(input_layers)):          
                if self.shape_not_compatible(reshape_size, input_layers[x]):
                    input_layers[x] = self.conform_layer_to_shape(reshape_size, input_layers[x])
                    print('conformed shape:')
                    print(input_layers[x]._keras_shape)
            
            layer = concatenate(input_layers) # should break here
        return layer
    

    def calculate_best_size_for_concatenated_layers(self, layers):
        
        """ 
        Rules for 'best' concatenation. If practical, reshape lower dimension inputs to 
        match higher dimension inputs (since high dimension inputs probably contain correlations
        between adjacents data points which we dont want to lose). This means that smaller
        image dimensions should be resized to match the larger image dimensions (and zero padded), and 1d data should
        be converted into a zero padded image
        
        Returns a shape (list) consisting of the maximum dimensions of the highest dimension layers
        """
        
        shape_details = [self.get_compiled_layer_shape_details(layer) for layer in layers]
        dimensions = [shape[0] for shape in shape_details]
        
        for x in range(len(dimensions)):
            dimensions[x] = list(filter(None, dimensions[x]))
        
        max_dimensions = max([len(d) for d in dimensions])
        highest_dimension_layers = [d for d in dimensions if len(d) == max_dimensions]
        
        shape_to_match = []
        for d in range(max_dimensions-1):
            edge_lengths = []
            for shapes in highest_dimension_layers:
                edge_lengths.append(shapes[d])
            shape_to_match.append(max(edge_lengths))
        
        shape_to_match.append(1)
        return shape_to_match

    
    def shape_not_compatible(self, shape, layer):
        
        return not self.shape_compatible(shape, layer)

    
    def shape_compatible(self, shape, layer):
        
        
        layer_shape = list(filter(None, layer._keras_shape))
    
        if layer_shape[:-1] == shape[:-1]:
            return True
        else:
            return False
            
    def conform_layer_to_shape(self, reshape_size, layer):
        
        """ 
        Reshapes and zero pads layers to make them fit the reshape_size. For 3d layers (images),
        each dimension in the reshape_size should be >= to the equivalent dimension in the 
        layer    
        """
        _, number_of_units_in_previous_layer, number_of_dimensions = self.get_compiled_layer_shape_details(layer)
        
        if number_of_dimensions == 1:
            layer = Reshape((number_of_units_in_previous_layer, 1))(layer)
            number_of_units_in_concatenation_dimensions = reduce(lambda x, y: x*y, reshape_size[:-1])
            
            padding_required = number_of_units_in_concatenation_dimensions - number_of_units_in_previous_layer%number_of_units_in_concatenation_dimensions
            
            layer = ZeroPadding1D((0, padding_required))(layer)
            _, padded_layer_size, _  = self.get_compiled_layer_shape_details(layer)
            
            reshape_to=reshape_size[:-1]
            reshape_to.append(int(padded_layer_size / number_of_units_in_concatenation_dimensions))
            layer = Reshape((reshape_to))(layer)
            
        if number_of_dimensions == 3:   
            layer_shape, _, _ = self.get_compiled_layer_shape_details(layer)
            layer_shape = list(filter(None, layer_shape))
            padding_required = []
            for x in range(len(layer_shape)-1):
                padding_required.append(reshape_size[x] - layer_shape[x])
        
            layer = ZeroPadding2D(((0, padding_required[0]), (0, padding_required[1])))(layer)
            
        
        return layer
    
    def get_compiled_layer_shape_details(self, layer):
        previous_layer_shape = layer._keras_shape
        number_of_units_in_previous_layer = reduce(lambda x, y: x*y,  [x for x in previous_layer_shape if x is not None])
        number_of_dimensions_in_previous_layer = len([x for x in previous_layer_shape if x is not None])
        
        return previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer


    def get_compiled_layer_for_id(self, layer_id):
        
        return [d['compiled_layer'] for d in self.list_of_trained_layers if d['layer_id'] == layer_id][0]
     
        
    def set_compiled_layer_for_id(self, layer_id, compiled_layer):   
        
        layer_details = [d for d in self.list_of_trained_layers if d['layer_id'] == layer_id][0]
        layer_details['compiled_layer'] = compiled_layer
    
        
    def get_reshape_size_closest_to_square(self, number_of_neurons, number_of_channels):
                
        return self.get_reshape_size_closest_to_aspect_ratio(number_of_neurons, number_of_channels, 1.0)
        
    
    def get_reshape_size_closest_to_aspect_ratio(self, number_of_neurons, number_of_channels, aspect_ratio):
        
        if type(number_of_neurons) != int:
            if not number_of_neurons.is_integer():
                raise ValueError('get_reshape_size_closest_to_aspect_ratio(number_of_neurons, number_of_channels, aspect_ratio). number_of_neurons should be an integer')
 
       
        aspect_ratio = float(aspect_ratio)
        
        neurons_per_channel = number_of_neurons / number_of_channels
        
        if not neurons_per_channel.is_integer():
            raise ValueError('get_reshape_size_closest_to_aspect_ratio: Non-integer number of neurons per channel')
 
        
        initial_dimension1 = int(math.sqrt(neurons_per_channel / aspect_ratio))
               
        step = 0
        while True:
            
            dimension1 = int(initial_dimension1 + step)            
            dimension2 = int(neurons_per_channel / dimension1)
            
            if self.calculate_number_of_neurons_in_shape((dimension1, dimension2, number_of_channels)) == number_of_neurons:
                return ((dimension1, dimension2, number_of_channels))
            
            dimension1 = int(initial_dimension1 - step)    
            
            if dimension1 == 0:
                return((1, neurons_per_channel, number_of_channels))
            
            dimension2 = int(neurons_per_channel / dimension1)
        
            if self.calculate_number_of_neurons_in_shape((dimension1, dimension2, number_of_channels)) == number_of_neurons:
                return ((dimension1, dimension2, number_of_channels))
            
            step = step + 1
        

    def get_checked_2d_kernel_size_for_layer(self, previous_layer_size, requested_kernel_size):
        
        kernel_size = []
        kernel_size.append(min([requested_kernel_size[0], previous_layer_size[1]]))
        kernel_size.append(min([requested_kernel_size[1], previous_layer_size[2]]))
        return kernel_size


    
    def calculate_number_of_neurons_in_shape(self, shape):
        
        return reduce(lambda x, y: x*y,  [x for x in shape if x is not None])
        
    def get_aspect_ratio(self, shape):
        
        if len(shape) > 3:
            aspect_ratio = shape[-3] / shape[-2]
        else:
            aspect_ratio = 1.0
        
        return aspect_ratio
        
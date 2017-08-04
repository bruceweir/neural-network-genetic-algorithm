"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

Original project:  https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
Extended by: bruce.weir@bbc.co.uk
"""
from keras.datasets import mnist, cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, Reshape, MaxPooling2D, Input, concatenate, ZeroPadding1D, ZeroPadding2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
from functools import reduce
import math
import sys
from ast import literal_eval
import numpy as np


K.set_image_dim_ordering('tf')

class Train():
# Helper: Early stopping.
    def __init__(self, kwargs):
        
        self.early_stopper = EarlyStopping(patience=5)
        self.list_of_trained_layers = []
        self.max_epochs = kwargs.get('max_epochs', sys.maxsize )
        self.dataset = kwargs.get('dataset', None)
        self.training_data_file = kwargs.get('training_data', None)
        self.test_data_file = kwargs.get('test_data', None)
        self.batch_size = kwargs.get('batch_size', 64)
        self.is_classification = kwargs.get('is_classification', True)
        
        if self.dataset == None and self.training_data_file == None:
            raise ValueError("""You need to specify either a dataset or training/test files to use.\n
Perhaps you should be launching the application from the command line? Example: python evolutionary_neural_network_generator.py -d mnist -p 10 -g 20""")
            
            
        if self.dataset == 'cifar10':
            self.get_cifar10()
        elif self.dataset == 'mnist':
            self.get_mnist()
        else:
            
            self.get_dataset_from_file(self.training_data_file, self.test_data_file)
            
            self.natural_input_shape = kwargs.get('natural_input_shape', None)                
            if self.natural_input_shape != None:
                self.natural_input_shape = literal_eval(self.natural_input_shape)
            else:
                self.natural_input_shape = self.input_shape

        print('Classification problem? ', self.is_classification)

    
    def get_cifar10(self):
        """Retrieve the CIFAR dataset and process the data."""
        # Set defaults.
        self.is_classification = True
        self.nb_classes = 10
        self.batch_size = 64
        self.input_shape = (3072,)
        self.natural_input_shape = (32, 32, 3)
        # Get the data.
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.x_train = self.x_train.reshape(50000, 3072)
        self.x_test = self.x_test.reshape(10000, 3072)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
    
        # convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, self.nb_classes)
        self.y_test = to_categorical(self.y_test, self.nb_classes)
    
        self.output_shape = (self.nb_classes,)
        

    def get_mnist(self):
        """Retrieve the MNIST dataset and process the data."""
        # Set defaults.
        self.is_classification = True
        self.nb_classes = 10
        self.batch_size = 128
        self.input_shape = (784,)
        # tensorflow-style ordering (Height, Width, Channels)
        self.natural_input_shape = (28, 28, 1)
    
        # Get the data.
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.reshape(60000, 784)
        self.x_test = self.x_test.reshape(10000, 784)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
    
        self.y_train = to_categorical(self.y_train, self.nb_classes)
        self.y_test = to_categorical(self.y_test, self.nb_classes)
    
        self.output_shape = (self.nb_classes,)
        

    def get_dataset_from_file(self, training_file_name, test_file_name):
        
        
        """ 
        Load the dataset from 2 numpy array save files. 'training_file_name' contains your training data,
        'test_file_name' contains your test data. The final column of each saved numpy array is the target
        output (which could contain numpy arrays). The other columns are the input vector.   
        
        Example: training data for an XOR gate
        
        [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        
        If the input has a natural shape (as an image for example), then set the --natural_input_shape
        value when starting the application.
        Example: If the input samples each form a 60x40x3 channel image, then the numpy array should
        be 7201 columns wide (the final column is the target output). The first 60 columns are the top
        row of the image and first 2400 columns are the first image channel.
        The --natural_input_shape should be "(60, 40, 3)"
            
        """
        if training_file_name == None or test_file_name == None:
            raise ValueError('Both a training data and a test data file needs to be specified.')
    
        print('Loading training data: %s' % training_file_name)
        
        training_data = np.load(training_file_name)
        
        self.x_train = training_data[:, range(training_data.shape[1]-1)]
        self.x_train = self.x_train.astype('float32')
        
        
        print('Loaded %d input training vectors, each of length: %d' % (self.x_train.shape[0], self.x_train.shape[1]))
        
        self.input_shape = (self.x_train.shape[1], )       
        print('Setting input_shape to: %s' % (self.input_shape,))
        
        self.y_train = training_data[:, training_data.shape[1]-1]
 
        print('Loading test data: %s' % test_file_name)        
        test_data = np.load(test_file_name)
        
        self.x_test = test_data[:, range(test_data.shape[1]-1)]
        self.x_test = self.x_test.astype('float32')
        
        self.y_test = test_data[:, test_data.shape[1]-1]

        print('Loaded %d input test vectors, each of length: %d' % (self.x_test.shape[0], self.x_test.shape[1]))

        if self.is_classification:
            self.nb_classes = len(np.unique(self.y_train))
        
            print('Found %d classes' % self.nb_classes)        
            print('Converting output values to categorical one-hot vectors')

            self.y_train = to_categorical(self.y_train, self.nb_classes)
            self.y_test = to_categorical(self.y_test, self.nb_classes)
 
        self.output_shape = list(np.array(self.y_train[0]).shape)
        
        self.y_train = np.array([a.reshape(self.output_shape) for a in self.y_train])
        self.y_test = np.array([a.reshape(self.output_shape) for a in self.y_test])
        
        print('Output data shape set to: ', self.output_shape)
        print('Output data training vector shape: ', self.y_train.shape)
        print('Output data test vector shape: ', self.y_test.shape)
        
        
    def train_and_score(self, network):
        """Train the model, store the accuracy in network.accuracy.
    
        Args:
            network: a Network object 
            
        """
           
        model = self.compile_model(network, self.output_shape, self.input_shape, self.natural_input_shape)
    
        if network.trained_model is None:
            network.trained_model = self.train_model(model, self.x_train, self.y_train, self.batch_size, self.max_epochs, self.x_test, self.y_test, [self.early_stopper])
            score = network.trained_model.evaluate(self.x_test, self.y_test, verbose=0)                
            network.loss = score[0]
            network.accuracy = score[1]
            
        print('Network training complete. Test accuracy: %f, Test Loss: %f' % (network.accuracy, network.loss))    

    def train_model(self, model, training_data, training_labels, batch_size, epochs, validation_data, validation_labels, callbacks=[]):
        
        model.fit(training_data, training_labels,
                      batch_size=batch_size,
                      epochs=epochs,  # using early stopping, so no real limit
                      verbose=True,
                      validation_data=(validation_data, validation_labels),
                      callbacks=callbacks)
        
        return model
    
    
    def compile_model(self, network, output_shape, input_shape, natural_input_shape):
        """Compile a sequential model.
    
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
        
        
        if self.is_classification:
        
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
        
        try:
            kernel_size = self.get_checked_2d_kernel_size_for_layer(previous_layer_shape, layer_parameters['kernel_size'])
            layer = Conv2D(layer_parameters['nb_filters'], 
                             kernel_size=kernel_size, 
                             strides=layer_parameters['strides'], 
                             activation=layer_parameters['activation'])(input_layer)
        except (ValueError, IndexError):
            print('add_conv2D_layer: Reshaping input')
            reshape_size = self.get_reshape_size_closest_to_square(number_of_units_in_previous_layer, natural_input_shape[-1])
            layer= Reshape(reshape_size)(input_layer)       
            layer = self.add_conv2D_layer(layer_parameters, layer, natural_input_shape);
            
        return layer

            
    def add_maxpooling2d_layer(self, layer_parameters, input_layer, natural_input_shape):
        
        previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer = self.get_compiled_layer_shape_details(input_layer)
        
        try:
                pool_size = self.get_checked_2d_kernel_size_for_layer(previous_layer_shape, layer_parameters['pool_size'])            
                layer = MaxPooling2D(pool_size=pool_size)(input_layer)
        
        except (ValueError, IndexError):
            reshape_size = self.get_reshape_size_closest_to_square(number_of_units_in_previous_layer, natural_input_shape[-1])
            layer= Reshape(reshape_size)(input_layer)
            layer = self.add_maxpooling2d_layer(layer_parameters, layer, natural_input_shape)
            
        return layer


    def add_dropout_layer(self, layer_parameters, input_layer):
        
        layer = Dropout(layer_parameters['remove_probability'])(input_layer)
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
                
        if type(number_of_neurons) != int:
            if not number_of_neurons.is_integer():
                raise ValueError('get_reshape_size_closest_to_square(number_of_neurons, number_of_channels). number_of_neurons should be an integer')
 
        dimension1 = math.sqrt(number_of_neurons / number_of_channels)
        
        dimension2 = number_of_neurons / (dimension1 * number_of_channels)
        
        dimension1 = float(math.ceil(dimension1))
        
        while dimension2.is_integer() is False:
            dimension1 -= 1
            dimension2 = number_of_neurons / dimension1
            
        closest_shape = (int(dimension1), int(dimension2), number_of_channels)
        
        total_size_of_closest_shape = reduce(lambda x, y: x*y,  [x for x in closest_shape])
        
        if total_size_of_closest_shape != number_of_neurons:
            return self.get_reshape_size_closest_to_square(number_of_neurons, 1)
        else:
            return (int(dimension1), int(dimension2), number_of_channels)
        

    def get_checked_2d_kernel_size_for_layer(self, previous_layer_size, requested_kernel_size):
        
        kernel_size = []
        kernel_size.append(min([requested_kernel_size[0], previous_layer_size[1]]))
        kernel_size.append(min([requested_kernel_size[1], previous_layer_size[2]]))
        return kernel_size


    
    def calculate_number_of_neurons_in_shape(self, shape):
        
        return reduce(lambda x, y: x*y,  [x for x in shape if x is not None])
        
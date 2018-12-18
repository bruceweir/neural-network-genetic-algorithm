"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

Original project:  https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
Extended by: bruce.weir@bbc.co.uk
"""
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.datasets import mnist, cifar10
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import backend as K
from network_compiler import Network_Compiler
import math
import sys
from ast import literal_eval
import numpy as np


#K.set_image_dim_ordering('tf')

class Train():
# Helper: Early stopping.
    def __init__(self, kwargs={}):
        
        self.early_stopper = EarlyStopping(patience=5)
        #self.list_of_trained_layers = []
        self.max_epochs = kwargs.get('max_epochs', sys.maxsize )
        self.dataset = kwargs.get('dataset', None)
        self.training_data_file = kwargs.get('training_data', None)
        self.test_data_file = kwargs.get('test_data', None)
        self.batch_size = kwargs.get('batch_size', 64)
        self.is_classification = kwargs.get('is_classification', True)
        self.network_compiler = Network_Compiler(kwargs)
        
        if self.dataset == None and self.training_data_file == None:
            raise ValueError("""You need to specify either a dataset or training/test files to use.\n
Perhaps you should be launching the application from the command line? Example: python evolutionary_neural_network_generator.py -d mnist -p 10 -g 20""")
            
            
        if self.dataset == 'cifar10':
            self.get_cifar10()
        elif self.dataset == 'mnist':
            self.get_mnist()
        else:        
            self.get_dataset_from_file(self.training_data_file, self.test_data_file)
            
        print('Classification problem? ', self.is_classification)

    
    def get_cifar10(self):
        """Retrieve the CIFAR dataset and process the data."""
        # Set defaults.
        self.is_classification = True
        self.nb_classes = 10
        self.batch_size = 64
        self.input_shape = (3072,)
        self.natural_input_shape = (32, 32, 3)
        self.natural_aspect_ratio = self.natural_input_shape[0] / self.natural_input_shape[1]

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
        self.natural_aspect_ratio = self.natural_input_shape[0] / self.natural_input_shape[1]
    
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
        'test_file_name' contains your test data. The first column of each saved numpy array is a numpy array
        consisting of the input tensor, the second column is another numpy array consisting of the target output tensor.
        
        Example: training data for an XOR gate
        
        xor_training_data = np.array([[np.array([0, 0], dtype=object), np.array([0], dtype=object)], 
                                      [np.array([0, 1], dtype=object), np.array([1], dtype=object)], 
                                      [np.array([1, 0], dtype=object), np.array([1], dtype=object)], 
                                      [np.array([1, 1], dtype=object), np.array([0], dtype=object)]], dtype=object)

    
        Example: Convert a 3x3 1 channel image of cross into a 3x3 1 channel image of a tick,
        and vice-versa
        
        cross = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=object).reshape((3, 3, 1))
        tick = np.array([[0, 0.1, .8], [0.6, .2, .6], [0, 0.8, .1]], dtype=object).reshape((3, 3, 1))
    
        training_data = np.array([[cross, tick],
                                  [tick, cross]])
    
            
        """
        if training_file_name == None or test_file_name == None:
            raise ValueError('Both a training data and a test data file needs to be specified.')
    
        print('Loading training data: %s' % training_file_name)
        
        training_data = np.load(training_file_name)
        
        self.x_train = training_data[:, 0]
        self.x_train = np.array([x.astype('float32') for x in self.x_train], dtype='object')
        
        
        print('Loaded %d input training vectors, each of length: %d' % (self.x_train.shape[0], self.x_train.shape[1]))
        
        self.input_shape = self.x_train.shape[1:]
        self.natural_input_shape = self.input_shape
        
        if(len(self.natural_input_shape) >= 3):
            self.natural_aspect_ratio = self.natural_input_shape[-3] / self.natural_input_shape[-2]
        else:
            self.natural_aspect_ratio = 1.0
        
        
        print('Setting input_shape to: %s' % (self.input_shape,))
        print('Input data aspect ratio: %f' % self.natural_aspect_ratio)
        
        self.y_train = training_data[:, 1]
        self.y_train = np.array([y.astype('float32') for y in self.y_train], dtype='object')
 
        print('Loading test data: %s' % test_file_name)        
        test_data = np.load(test_file_name)
        
        self.x_test = test_data[:, 0]
        self.x_test = np.array([x.astype('float32') for x in self.x_test], dtype='object')
        
        self.y_test = test_data[:, 1]
        self.y_test = np.array([y.astype('float32') for y in self.y_test], dtype='object')
 
        print('Loaded %d input test vectors, each of length: %d' % (self.x_test.shape[0], self.x_test.shape[1]))

        if self.is_classification:
            self.nb_classes = len(np.unique(self.y_train))
        
            print('Found %d classes' % self.nb_classes)        
            print('Converting output values to categorical one-hot vectors')

            self.y_train = to_categorical(self.y_train, self.nb_classes)
            self.y_test = to_categorical(self.y_test, self.nb_classes)
 
        self.output_shape = tuple(np.array(self.y_train[0]).shape)
        
        print('Output data shape set to: ', self.output_shape)
        print('Output data test vector shape: ', self.y_test.shape)
        
        
        
    def train_and_score(self, network):
        """Train the model, store the accuracy in network.accuracy.
    
        Args:
            network: a Network object 
            
        """
        
        model = self.network_compiler.compile_model(network, self.output_shape, self.input_shape, self.natural_input_shape, self.is_classification)
    
        if network.trained_model is None:
            network.trained_model = self.train_model(model, self.x_train, self.y_train, self.batch_size, self.max_epochs, self.x_test, self.y_test, [self.early_stopper])
            if network.trained_model is not None:
                score = network.trained_model.evaluate(self.x_test, self.y_test, verbose=0)                
                network.loss = score[0]
                network.accuracy = score[1]
            else:
                network.loss = math.inf
                network.accuracy = 0
        
        print('Network training complete. Test accuracy: %f, Test Loss: %f' % (network.accuracy, network.loss))    

    def train_model(self, model, training_data, training_labels, batch_size, epochs, validation_data, validation_labels, callbacks=[]):
        
        try:
            model.fit(training_data, training_labels,
                          batch_size=batch_size,
                          epochs=epochs,  # using early stopping, so no real limit
                          verbose=True,
                          validation_data=(validation_data, validation_labels),
                          callbacks=callbacks)
        except Exception as e:
            print(e)
            if batch_size/2 > 1:
                print('Model untrainable, trying with reduced batch_size')
                model = self.train_model(model, training_data, training_labels, int(batch_size/2), epochs, validation_data, validation_labels)
            else:
                model = None
            
        return model
    
    

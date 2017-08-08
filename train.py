"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

Original project:  https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
Extended by: bruce.weir@bbc.co.uk
"""
from keras.datasets import mnist, cifar10
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
from network_compiler import Network_Compiler
import math
import sys
from ast import literal_eval
import numpy as np


K.set_image_dim_ordering('tf')

class Train():
# Helper: Early stopping.
    def __init__(self, kwargs):
        
        self.early_stopper = EarlyStopping(patience=5)
        #self.list_of_trained_layers = []
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
                if(len(self.natural_input_shape) > 1):
                    self.natural_aspect_ratio = self.natural_input_shape[0] / self.natural_input_shape[1]
                else:
                    self.natural_aspect_ratio = 1.0
            else:
                self.natural_input_shape = self.input_shape
                self.natural_aspect_ratio = 1.0

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
        'test_file_name' contains your test data. The final column of each saved numpy array is the target
        output (which could contain numpy arrays). The other columns are the input vector.   
        
        Example: training data for an XOR gate
        
        xor = np.array([[0, 0, np.array([0], dtype=object)], 
                    [0, 1, np.array([1], dtype=object)], 
                    [1, 0, np.array([1], dtype=object)], 
                    [1, 1, np.array([0], dtype=object)]], dtype=object)
    
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
        network_compiler = Network_Compiler()
        
        model = network_compiler.compile_model(network, self.output_shape, self.input_shape, self.natural_input_shape, self.is_classification)
    
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
    
    
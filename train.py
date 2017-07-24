"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, Reshape, MaxPooling2D, Input
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import sys
from functools import reduce
import math

K.set_image_dim_ordering('tf')


# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)
    input_shape_conv2d = (32, 32, 3)
    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, input_shape_conv2d)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)
    # tensorflow-style ordering (Height, Width, Channels)
    input_shape_conv2d = (28, 28, 1)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #x_train_conv2d = x_train.reshape(60000, 28, 28, 1)
    #x_test_conv2d = x_test.reshape(10000, 28, 28, 1)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, input_shape_conv2d)


def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test, input_shape_conv2d = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test, input_shape_conv2d = get_mnist()


    model = compile_model(network, nb_classes, input_shape, input_shape_conv2d)

    if network.trained_model is None:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=10000,  # using early stopping, so no real limit
                  verbose=True,
                  validation_data=(x_test, y_test),
                  callbacks=[early_stopper])

        network.trained_model = model
    
    score = network.trained_model.evaluate(x_test, y_test, verbose=0)

    
    return score[1]  # 1 is accuracy. 0 is loss.

def compile_model(network, nb_classes, input_shape, input_shape_conv2d):
    """Compile a sequential model.

    Args:
        network_layers (dict): the parameters of the network inbetween the input and output layer

    Returns:
        a compiled network.

    """
    
    # Get our network parameters.
    
    inputs = Input(shape=input_shape)
    previous_layer_shape = inputs._keras_shape
    
    if network.starts_with_2d_layer() and len(input_shape) == 1:
        layer = Reshape(input_shape_conv2d)(inputs)
        previous_layer_shape = layer._keras_shape
    else:
        layer=inputs
            

    number_of_units_in_previous_layer = reduce(lambda x, y: x*y,  [x for x in previous_layer_shape if x is not None])
    # Add each layer.
    for i in range(network.number_of_layers()):
        
        layer_type = network.get_network_layer_type(i);
        layer_parameters = network.get_network_layer_parameters(i)
                
        # Need input shape for first layer.
        if i == 0 and layer_type == 'Dense':
            layer = Dense(layer_parameters['nb_neurons'], 
                                activation=layer_parameters['activation'])(layer)
                
        else:
            if layer_type == 'Dense':
                layer = Dense(layer_parameters['nb_neurons'], 
                                activation=layer_parameters['activation'])(layer)
                
            elif layer_type == 'Conv2D':
                kernel_size = get_checked_2d_kernel_size_for_layer(previous_layer_shape, layer_parameters['kernel_size'])
                
                
                layer = Conv2D(layer_parameters['nb_filters'], 
                                 kernel_size=kernel_size, 
                                 strides=layer_parameters['strides'], 
               #                  padding='same',
                                 activation=layer_parameters['activation'])(layer)
                
            elif layer_type == 'MaxPooling2D':
                pool_size = get_checked_2d_kernel_size_for_layer(previous_layer_shape, layer_parameters['pool_size'])
                layer = MaxPooling2D(pool_size=pool_size)(layer)
                
            elif layer_type == 'Dropout':
                layer = Dropout(layer_parameters['remove_probability'])(layer)
                
            elif layer_type == 'Flatten':
                layer = Flatten()(layer)
                #dont get previous layer shape from here

            elif layer_type == 'Reshape':                                     
                layer_reshape_factor = layer_parameters['first_dimension_scale']
                reshape_dimensions = get_closest_valid_reshape_for_given_scale(number_of_units_in_previous_layer, layer_reshape_factor)               
                layer = Reshape((reshape_dimensions[0], reshape_dimensions[1], 1))(layer)
                
        
        #if len([x for x in layer.shape.as_list() if x is not None]) > 0:
        previous_layer_shape = layer._keras_shape
        number_of_units_in_previous_layer = reduce(lambda x, y: x*y,  [x for x in previous_layer_shape if x is not None])
            
                
        

    # Output layer.
    if network.network_is_not_1d_at_layer(len(network.network_layers)-1):
        layer = Flatten()(layer)
    
    predictions = Dense(nb_classes, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=predictions)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    #SVG(model_to_dot(model).create(prog='dot', format='svg'))

    return model


def create_test_model():
    model = Sequential()

    model.add(Dense(512, input_shape=(784,)))
    model.add(Reshape((32, 16, 1)))
    model.add(Conv2D(64, (5,5), strides=(2,2)))#, padding='same'))
    print(model.get_layer(index=-1).output_shape)
    
    
def get_closest_valid_reshape_for_given_scale(number_of_neurons, reshape_factor):
    
    if reshape_factor > number_of_neurons:
        return (number_of_neurons, 1)
    
    while (number_of_neurons/reshape_factor).is_integer() is False and reshape_factor <= number_of_neurons:
        reshape_factor = reshape_factor+1
    
    return(int(number_of_neurons/reshape_factor), reshape_factor)

def get_checked_2d_kernel_size_for_layer(previous_layer_size, requested_kernel_size):
    
    kernel_size = []
    kernel_size.append(min([requested_kernel_size[0], previous_layer_size[1]]))
    kernel_size.append(min([requested_kernel_size[1], previous_layer_size[2]]))
    return kernel_size



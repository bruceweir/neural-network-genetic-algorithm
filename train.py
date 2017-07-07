"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot



K.set_image_dim_ordering('tf')

import json

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

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

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

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

    x_train_conv2d = x_train.reshape(60000, 28, 28)
    x_test_conv2d = x_test.reshape(10000, 28, 28)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, input_shape_conv2d, x_train_conv2d, x_test_conv2d)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network_layers (dict): the parameters of the network inbetween the input and output layer

    Returns:
        a compiled network.

    """

    print(json.dumps(network.network_layers, indent=4))
    # Get our network parameters.
    nb_layers = len(network.network_layers)

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        print(i)
        layer_type = network.network_layers[i]['layer_type'];
        layer_parameters = network.network_layers[i]['layer_parameters']
        # Need input shape for first layer.
        if i == 0:
            if layer_type == 'Dense':
                model.add(Dense(layer_parameters['nb_neurons'], activation=layer_parameters['activation'], input_shape=input_shape))
            elif layer_type == 'Conv2D':
                print('Conv2D params')
                print(layer_parameters['nb_filters'])
                print(layer_parameters['kernel_size'])
                print(input_shape)
                model.add(Conv2D(layer_parameters['nb_filters'], kernel_size=layer_parameters['kernel_size'], input_shape=input_shape))#, activation=layer_parameters['activation'], input_shape=input_shape))       
                
        else:
            if layer_type == 'Dense':
                model.add(Dense(layer_parameters['nb_neurons'], activation=layer_parameters['activation']))
            elif layer_type == 'Conv2D':
                print('Conv2D params')                
                print(layer_parameters['nb_filters'])
                print(layer_parameters['kernel_size'])
                
                #model.add(Conv2D(layer_parameters['nb_filters'], kernel_size=layer_parameters['kernel_size'], strides=layer_parameters['strides'], activation=layer_parameters['activation']))       
                model.add(Conv2D(layer_parameters['nb_filters'], kernel_size=layer_parameters['kernel_size']))#, strides=layer_parameters['strides'], activation=layer_parameters['activation']))       
            elif layer_type == 'Dropout':
                model.add(Dropout(layer_parameters['remove_probability']))
            elif layer_type == 'Flatten':
                model.add(Flatten())
            
        

    # Output layer.
    if network.network_is_not_1d_at_layer(len(network.network_layers)-1):
        model.add(Flatten())
    
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    #SVG(model_to_dot(model).create(prog='dot', format='svg'))

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test, input_shape_conv2d, x_train_conv2d, x_test_conv2d = get_mnist()

    input_shape_choice = input_shape
    x_train_choice = x_train
    x_test_choice = x_test
    
    if network.starts_with_2d_layer():
        input_shape_choice = input_shape_conv2d
        x_train_choice = x_train_conv2d
        x_test_choice = x_test_conv2d
        
    model = compile_model(network, nb_classes, input_shape_choice)

    model.fit(x_train_choice, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test_choice, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test_choice, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.

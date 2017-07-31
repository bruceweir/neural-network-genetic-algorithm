"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, Reshape, MaxPooling2D, Input, concatenate, ZeroPadding1D, ZeroPadding2D
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

list_of_trained_layers = []

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
        network.trained_model = train_model(model, x_test, y_test, batch_size, 10000, x_test, y_test, [early_stopper])
        
                
    score = network.trained_model.evaluate(x_test, y_test, verbose=0)

    
    return score[1]  # 1 is accuracy. 0 is loss.

def train_model(model, training_data, training_labels, batch_size, epochs, validation_data, validation_labels, callbacks=[]):
    
    model.fit(training_data, training_labels,
                  batch_size=batch_size,
                  epochs=epochs,  # using early stopping, so no real limit
                  verbose=True,
                  validation_data=(validation_data, validation_labels),
                  callbacks=callbacks)
    
    return model
    
    
def compile_model(network, nb_classes, input_shape, input_shape_conv2d):
    """Compile a sequential model.

    Args:
        network: A network object with a defined network structure
        nb_classes: The number of classification classes to output
        input_shape: Shape of the input vector (i.e. for training on MNIST: (784,))
        input_shape_conv2d: Shape of the input vector if it can be sensibly considered as a 2D image (i.e. for training on greyscale MNIST only: (28,28,1))
        

    Returns:
        a compiled Keras Model which can be trained.

    """
    
    global list_of_trained_layers
    list_of_trained_layers = [{'layer_id':x, 'compiled_layer':None} for x in network.get_all_network_layer_ids()]
    # Get our network parameters.
    
    inputs = Input(shape=input_shape)
    layer = inputs

    first_layer_ids = network.get_network_layers_with_no_upstream_connections()
    if len(first_layer_ids) != 1:
        # For now, connect the layers - TODO, deal with multiple inputs case
        network.connect_layers([first_layer_ids[0]], first_layer_ids[1:])
    
    last_layer_ids = network.get_network_layers_with_no_downstream_connections()
    if len(last_layer_ids) != 1:
        # For now, connect the layers - TODO, deal with multiple outputs case        
        network.connect_layers(last_layer_ids[1:], [last_layer_ids[0]])
    
    final_layer_id = last_layer_ids[0]
    
    layer = add_layer(network, final_layer_id, layer)
   
    _, _, number_of_dimensions_in_previous_layer = get_compiled_layer_shape_details(layer)
    
    if number_of_dimensions_in_previous_layer > 2:
        layer = Flatten()(layer)
        
    predictions = add_dense_layer({'activation': 'softmax', 'nb_neurons': nb_classes}, layer)

    model = Model(inputs=inputs, outputs=predictions, name='Output')
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])


    return model

def add_layer(network, layer_id, input_layer):
    
    global list_of_trained_layers
    """ Starting at the output layer, this should recurse up the network graph, adding Keras layers """
    
    if get_compiled_layer_for_id(layer_id) is not None:
        return get_compiled_layer_for_id(layer_id)
    
    
    layer = input_layer
    
    layers_input_into_this_level = []
    
    for ids in network.get_upstream_layers(layer_id):
        layers_input_into_this_level.append(add_layer(network, ids, input_layer))

    
    if len(layers_input_into_this_level) > 1:
        layer = add_concatenation(layers_input_into_this_level)       
    
    elif len(layers_input_into_this_level) == 1:        
        layer = layers_input_into_this_level[0]

    layer_type = network.get_network_layer_type(layer_id)
    layer_parameters = network.get_network_layer_parameters(layer_id)

    
 
    if layer_type == 'Dense':
        layer = add_dense_layer(layer_parameters, layer)
        
    elif layer_type == 'Conv2D':            
        layer = add_conv2D_layer(layer_parameters, layer)            
            
    elif layer_type == 'MaxPooling2D':
        layer = add_maxpooling2d_layer(layer_parameters, layer)
            
    elif layer_type == 'Dropout':
        layer = Dropout(layer_parameters['remove_probability'])(layer)
        
    
    previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer = get_compiled_layer_shape_details(layer)

    set_compiled_layer_for_id(layer_id, layer)
    
    return layer
    
    
def add_dense_layer(layer_parameters, input_layer):
    
    try:
        layer = Dense(layer_parameters['nb_neurons'], 
                      activation=layer_parameters['activation'])(input_layer)
    except:
        print('add_dense_layer: Flattening input')
        layer = Flatten()(input_layer)
        layer = Dense(layer_parameters['nb_neurons'], 
                  activation=layer_parameters['activation'])(layer)
    return layer

    
def add_conv2D_layer(layer_parameters, input_layer):
    
    previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer = get_compiled_layer_shape_details(input_layer)
    
    try:
        kernel_size = get_checked_2d_kernel_size_for_layer(previous_layer_shape, layer_parameters['kernel_size'])
        layer = Conv2D(layer_parameters['nb_filters'], 
                         kernel_size=kernel_size, 
                         strides=layer_parameters['strides'], 
                         activation=layer_parameters['activation'])(input_layer)
    except (ValueError, IndexError):
        print('add_conv2D_layer: Reshaping input')
        reshape_size = get_reshape_size_closest_to_square(number_of_units_in_previous_layer)
        layer= Reshape((reshape_size[0], reshape_size[1], 1))(input_layer)       
        layer = add_conv2D_layer(layer_parameters, layer);
        
    return layer
            
def add_maxpooling2d_layer(layer_parameters, input_layer):
    
    previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer = get_compiled_layer_shape_details(input_layer)
    
    try:
            pool_size = get_checked_2d_kernel_size_for_layer(previous_layer_shape, layer_parameters['pool_size'])            
            layer = MaxPooling2D(pool_size=pool_size)(input_layer)
    
    except (ValueError, IndexError):
        reshape_size = get_reshape_size_closest_to_square(number_of_units_in_previous_layer)
        layer= Reshape((reshape_size[0], reshape_size[1], 1))(input_layer)
        layer = add_maxpooling2d_layer(layer_parameters, layer)
        
    return layer

def add_concatenation(input_layers):
    
    try:
        layer = concatenate(input_layers)
    except ValueError:
        
        reshape_size = calculate_best_size_for_concatenated_layers(input_layers)
        
        for x in range(len(input_layers)):          
            if shape_not_compatible(reshape_size, input_layers[x]):
                input_layers[x] = conform_layer_to_shape(reshape_size, input_layers[x])
                print('conformed shape:')
                print(input_layers[x]._keras_shape)
        
        layer = concatenate(input_layers) # should break here
    return layer
    

def calculate_best_size_for_concatenated_layers(layers):
    
    """ 
    Rules for 'best' concatenation. If practical, reshape lower dimension inputs to 
    match higher dimension inputs (since high dimension inputs probably contain correlations
    between adjacents data points which we dont want to lose). This means that smaller
    image dimensions should be resized to match the larger image dimensions (and zero padded), and 1d data should
    be converted into a zero padded image
    
    Returns a shape (list) consisting of the maximum dimensions of the highest dimension layers
    """
    
    shape_details = [get_compiled_layer_shape_details(layer) for layer in layers]
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

#    longest_dimensions_except_last = [sum(x[:-1]) for x in highest_dimension_layers]
#    index_of_target_shape = longest_dimensions_except_last.index(max(longest_dimensions_except_last))
#    shape_to_match_to = highest_dimension_layers[index_of_target_shape] 
    
#    return shape_to_match_to

def shape_not_compatible(shape, layer):
    
    return not shape_compatible(shape, layer)


def shape_compatible(shape, layer):
    
    
    layer_shape = list(filter(None, layer._keras_shape))

    print('shape: %s, layer_shape: %s' % (shape, layer_shape))
    if layer_shape[:-1] == shape[:-1]:
        return True
    else:
        return False
        
def conform_layer_to_shape(reshape_size, layer):
    
    """ 
    Reshapes and zero pads layers to make them fit the reshape_size. For 3d layers (images),
    each dimension in the reshape_size should be >= to the equivalent dimension in the 
    layer    
    """
    _, number_of_units_in_previous_layer, number_of_dimensions = get_compiled_layer_shape_details(layer)
    
    if number_of_dimensions == 1:
        layer = Reshape((number_of_units_in_previous_layer, 1))(layer)
        number_of_units_in_concatenation_dimensions = reduce(lambda x, y: x*y, reshape_size[:-1])
        #print(number_of_units_in_concatenation_dimensions)
        #print(number_of_units_in_previous_layer)
        padding_required = number_of_units_in_concatenation_dimensions - number_of_units_in_previous_layer%number_of_units_in_concatenation_dimensions
        #print('padding reqd = %d' % padding_required)
        layer = ZeroPadding1D((0, padding_required))(layer)
        _, padded_layer_size, _  = get_compiled_layer_shape_details(layer)
        #print('padded_layer_size: %d' % padded_layer_size)
        
        reshape_to=reshape_size[:-1]
        reshape_to.append(int(padded_layer_size / number_of_units_in_concatenation_dimensions))
        layer = Reshape((reshape_to))(layer)
        
    if number_of_dimensions == 3:   
        layer_shape, _, _ = get_compiled_layer_shape_details(layer)
        layer_shape = list(filter(None, layer_shape))
        padding_required = []
        for x in range(len(layer_shape)-1):
            padding_required.append(reshape_size[x] - layer_shape[x])
    
        layer = ZeroPadding2D(((0, padding_required[0]), (0, padding_required[1])))(layer)
        
    
    return layer
    
def get_compiled_layer_shape_details(layer):
    previous_layer_shape = layer._keras_shape
    number_of_units_in_previous_layer = reduce(lambda x, y: x*y,  [x for x in previous_layer_shape if x is not None])
    number_of_dimensions_in_previous_layer = len([x for x in previous_layer_shape if x is not None])
    
    return previous_layer_shape, number_of_units_in_previous_layer, number_of_dimensions_in_previous_layer


def get_compiled_layer_for_id(layer_id):
    
    global list_of_trained_layers
    return [d['compiled_layer'] for d in list_of_trained_layers if d['layer_id'] == layer_id][0]
 
def set_compiled_layer_for_id(layer_id, compiled_layer):   
    global list_of_trained_layers
    
    layer_details = [d for d in list_of_trained_layers if d['layer_id'] == layer_id][0]
    layer_details['compiled_layer'] = compiled_layer

    
def get_reshape_size_closest_to_square(number_of_neurons):
    
    dimension1 = math.sqrt(number_of_neurons)
    dimension2 = number_of_neurons / dimension1
    
    dimension1 = float(math.ceil(dimension1))
    
    while dimension2.is_integer() is False:
        dimension1 -= 1
        dimension2 = number_of_neurons / dimension1
        
    
    return (int(dimension1), int(dimension2))
        

def get_checked_2d_kernel_size_for_layer(previous_layer_size, requested_kernel_size):
    
    kernel_size = []
    kernel_size.append(min([requested_kernel_size[0], previous_layer_size[1]]))
    kernel_size.append(min([requested_kernel_size[1], previous_layer_size[2]]))
    return kernel_size



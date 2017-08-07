# Evolve a neural network with a genetic algorithm

This application uses a genetic algorithm to search for an optimum neural network structure.

It can currently handle Dense (Fully Connected), Conv2D, Dropout and MaxPooling layers and branching neural network structures. It uses the Keras library to build, train and validate.

For more, see this blog post, which describes the original code from which this is forked: 
https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164

## To run

To run the genetic algorithm:

```python3 evolutionary_neural_network_generator.py```

Options are viewable via the -h switch


To Run the Unit Tests:

```python3 tests.py```

You can set your network parameter choices by editing network_layer_options.py.

You can also choose whether to use the MNIST or CIFAR10 datasets. Simply set `-dataset` switch to either `mnist` or `cifar10`. You can also use your own datasets by following the structure of the get_mnist() or get_cifar10() functions.

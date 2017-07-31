# Evolve a neural network with a genetic algorithm

This is an example of how we can use a genetic algorithm in an attempt to find the optimal network parameters for classification tasks.

It can currently handle Dense, Conv2D, Dropout and MaxPooling layers and branching neural network structures. It uses the Keras library to build, train and validate.

On the easy MNIST dataset, we are able to quickly find a network that reaches > 98% accuracy. On the more challenging CIFAR10 dataset, we get to 56% after 10 generations (with population 20).

For more, see this blog post, which describes the original code from which this is forked: 
https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164

## To run

To run the genetic algorithm:

```python3 evolutionary_neural_network_generator.py```

Options are viewable via the -h switch


To Run the Unit Tests:

```python3 tests.py```

You can set your network parameter choices by editing network_layer_options.py.

You can also choose whether to use the MNIST or CIFAR10 datasets. Simply set `-dataset` switch to either `mnist` or `cifar10`. You can add your own datasets by following the structure of the get_mnist() or get_cifar10() functions.

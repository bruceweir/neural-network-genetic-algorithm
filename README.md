# Evolve a neural network with a genetic algorithm

This is an example of how we can use a genetic algorithm in an attempt to find the optimal network parameters for classification tasks.

It's currently limited to only Sequential networks (ie. one input and one output tensor per layer, with Fully Connected (Dense) and 2D Convolutional layers) and uses the Keras library to build, train and validate.

On the easy MNIST dataset, we are able to quickly find a network that reaches > 98% accuracy. On the more challenging CIFAR10 dataset, we get to 56% after 10 generations (with population 20).

For more, see this blog post: 
https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164

## To run

To run the genetic algorithm:

```python3 evolutionary_neural_network_generator.py```

Options are viewable via the -h switch


To Run the Unit Tests:

```python3 tests.py```

You can set your network parameter choices by editing network_layer_options.py. You can also choose whether to use the MNIST or CIFAR10 datasets. Simply set `-dataset` to either `mnist` or `cifar10`.

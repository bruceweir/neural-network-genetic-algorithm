# Evolve a neural network with a genetic algorithm

This application uses a genetic algorithm to search for an optimum neural network structure.

It can currently handle Dense (Fully Connected), Conv2D, Dropout, MaxPooling and ActivityRegularization layers and branching neural network structures. It uses the Keras library to build, train and validate.

For more, see this blog post, which describes the original code from which this is forked: 
https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164

## To run

To run the genetic algorithm:

```python3 evolutionary_neural_network_generator.py```

Options are viewable via the -h switch

Example usage for running breeding MNIST classifiers using a population of 10 networks over 20 generations:
(Note that if you don't already have MNIST installed it will be downloaded for you, which might not work if
 you are behind a proxy)

```python3 evolutionary_neural_network_generator.py -p 10 -g 20 -d mnist```

To Run the Unit Tests:

```python3 tests.py```

You can set your network parameter choices by editing network_layer_options.py.

You can choose whether to use the MNIST or CIFAR10 datasets. Simply set `-dataset` switch to either `mnist` or `cifar10`. 
You can also use your own datasets by using the --training_data and --test_data switches and following the data formatting instructions in the get_dataset_from_file() method in train.py,
or the examples in the test_train() function in tests.py


"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
from network import Network
import copy

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, retain=0.4,
                 random_select=0.1, mutate_chance=0.2, forbidden_layer_types=[]):
        """Create an optimizer.

        Args:
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.forbidden_layer_types = forbidden_layer_types
        

    def create_population(self, count, initial_length = 1):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        
        if count < 2:
            print('Minimum population count is 2. So using that.')
            count = 2
            
        population = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.forbidden_layer_types)
            network.create_random_network(initial_length, True)

            # Add the network to our population.
            population.append(network)

        return population

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, population):
        """Find average fitness for a population.

        Args:
            population (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(network) for network in population))
        return summed / float((len(population)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        
        
            
        children = []
        for _ in range(2):

            child = Network()

            if father.number_of_layers() > mother.number_of_layers():               
                longest_network = father
                shortest_network = mother
            else:
                longest_network = mother
                shortest_network = father               
                
            for i in range(shortest_network.number_of_layers()):
                    child.network_layers.append(copy.deepcopy(random.choice([shortest_network.network_layers[i], longest_network.network_layers[i]])))
            for i in range(longest_network.number_of_layers() - shortest_network.number_of_layers()):
                    if random.random() > 0.5:
                        child.network_layers.append(copy.deepcopy(longest_network.network_layers[i + shortest_network.number_of_layers()]))
            
            child.check_network_structure()
            
            children.append(child)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        if len(network.network_layers) > 1:         
            mutationType = random.choice(['AdjustLayerParameter', 'RemoveLayer', 'InsertLayer'])
        else:
            mutationType = random.choice(['AdjustLayerParameter', 'InsertLayer'])
            
        
        
        mutatedLayerIndex = random.choice(range(len(network.network_layers)))
        mutatedLayerType = network.get_network_layer_type(mutatedLayerIndex)
        
        print('Mutating network: %s. Index: %d (%s)' % (mutationType, mutatedLayerIndex, mutatedLayerType))
        # Mutate one of the params.
        if mutationType == 'AdjustLayerParameter':
            if mutatedLayerType != 'Flatten':
                parameter, value = self.get_random_parameter_for_network_layer(network, mutatedLayerIndex)
                network.get_network_layer_parameters(mutatedLayerIndex)[parameter] = value
        elif mutationType == 'RemoveLayer':
            del network.network_layers[mutatedLayerIndex]
        elif mutationType == 'InsertLayer':
            allow_dropout = False
            if len(network.network_layers) > 1 and network.get_network_layer_type(mutatedLayerIndex-1) != 'Dropout':
                allow_dropout = True
            
            network.network_layers.insert(mutatedLayerIndex, network.create_random_layer(allow_dropout = allow_dropout))

        network.check_network_structure()
                
        return network

    def get_random_parameter_for_network_layer(self, network, layer_index):
        
        network_layer = network.network_layers[layer_index]
        layer_type = network_layer['layer_type']
    
        if layer_type == 'Dense':
            parameter = random.choice(list(network.get_dense_layer_options().keys()))
            value = random.choice(network.get_dense_layer_options()[parameter])
        elif layer_type == 'Conv2D':
            parameter = random.choice(list(network.get_conv2d_layer_options().keys()))
            value = random.choice(network.get_conv2d_layer_options()[parameter])
        elif layer_type == 'Dropout':
            parameter = random.choice(list(network.get_dropout_layer_options().keys()))
            value = random.choice(network.get_dropout_layer_options()[parameter])
        elif layer_type == 'Reshape':
            parameter = random.choice(list(network.get_reshape_layer_options().keys()))
            value = random.choice(network.get_reshape_layer_options()[parameter])
        elif layer_type == 'MaxPooling2D':
            parameter = random.choice(list(network.get_maxpooling2d_layer_options().keys()))
            value = random.choice(network.get_maxpooling2d_layer_options()[parameter])

        else:
            raise NameError('Error: unknown layer_type: %s' % layer_type)
            
        return parameter, value
    
    def evolve(self, population):
        """Evolve a population of networks.

        Args:
            population (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in population]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen (with a minimum of 2).
        retain_length = max([int(len(graded)*self.retain), 2])

        # The parents are every network we want to keep.       
        parents = graded[:retain_length]

        networks_to_delete = []
        
        # For those we aren't keeping, randomly keep some anyway, otherwise 
        # add them to the networks_to_delete array.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
            else:
                networks_to_delete.append(individual)

        # this clean up is probably necessary since the networks now keep
        # a copy of their trained network, which might be very large
        while len(networks_to_delete):
            del networks_to_delete[0]
        

        # Randomly mutate some of the networks we're keeping.
        for individual in parents:
            if self.mutate_chance > random.random():
                individual = self.mutate(individual)
            
        print('mutations complete')
        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            print('breeding children')
            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents

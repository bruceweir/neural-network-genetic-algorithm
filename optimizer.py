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

    def __init__(self, **kwargs):
        """Create an optimizer.

        Args:
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = kwargs.get('mutate_chance', 0.2)
        self.random_select = kwargs.get('random_select', 0.1)
        self.retain = kwargs.get('retain', 0.4)
        self.forbidden_layer_types = kwargs.get('forbidden_layer_types', [])
        self.elitist = kwargs.get('elitist', False)
        #self.population_size = kwargs['population']
        #self.initial_network_length = kwargs['initial_network_length']
        
        

    def create_population(self, population_size, initial_network_length=1):
        """Create a population of random networks.

        Args:
            population_size (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        
        
        if population_size < 2:
            print('Minimum population count is 2. So using that.')
            population_size = 2
            
        population = []
        for _ in range(0, population_size):
            # Create a random network.
            network = Network(self.forbidden_layer_types)
            network.create_random_network(initial_network_length, True)

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


    
    def evolve(self, population):
        """Evolve a population of networks.

        Args:
            population (list): A list of networks

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
        first_parent_to_mutate = 0
        
        if self.elitist is True:
            first_parent_to_mutate = 1
            
        for individual in parents[first_parent_to_mutate:]:
            if self.mutate_chance > random.random():
                self.mutate(individual)
            
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
    
    
    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network: A network object to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        if network.number_of_layers() > 1:         
            mutation_type = random.choice(['AdjustLayerParameter', 'RemoveLayer', 'InsertLayer'])
        else:
            mutation_type = random.choice(['AdjustLayerParameter', 'InsertLayer'])
            
        
        
        mutated_layer_id = random.choice(range(network.number_of_layers()))
        mutated_layer_type = network.get_network_layer_type(mutated_layer_id)
        
        print('Mutating network: %s. Index: %d (%s)' % (mutation_type, mutated_layer_id, mutated_layer_type))
        # Mutate one of the params.
        if mutation_type == 'AdjustLayerParameter':
            self.mutate(network)
        elif mutation_type == 'RemoveLayer':           
            network.delete_layer(mutated_layer_id)
        elif mutation_type == 'InsertLayer':
            network.insert_random_layer(mutated_layer_id)

                
        #return network

    
    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        
            
        babies = []
        for _ in range(2):

            baby = Network(self.forbidden_layer_types)

            if father.number_of_layers() > mother.number_of_layers():               
                longest_network = father
                shortest_network = mother
            else:
                longest_network = mother
                shortest_network = father               
                
            for i in range(shortest_network.number_of_layers()):
                    baby.network_layers.append(copy.deepcopy(random.choice([shortest_network.network_layers[i], longest_network.network_layers[i]])))
            for i in range(longest_network.number_of_layers() - shortest_network.number_of_layers()):
                    if random.random() > 0.5:
                        baby.network_layers.append(copy.deepcopy(longest_network.network_layers[i + shortest_network.number_of_layers()]))
            
            
            babies.append(baby)

        return babies

    
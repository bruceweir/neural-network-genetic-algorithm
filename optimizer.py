"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
    Original project:  https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
    Extended by: bruce.weir@bbc.co.uk
"""
from functools import reduce
from operator import add
import random
from network import Network
import copy

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, kwargs):
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
        self.is_classification = kwargs.get('is_classification', False)
        
        

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
            network.create_random_network(initial_network_length)

            # Add the network to our population.
            population.append(network)

        return population

    def fitness(self, network):
        """Return the fitness appropriate to the problem."""
        if self.is_classification:
            return network.accuracy
        else:
            return network.loss

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

        # Sort on the fitness.
        reverse = False
        
        if self.is_classification:
            reverse = True
            
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=reverse)]

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
            mutation_type = random.choice(['AdjustLayerParameter', 'RemoveLayer', 'InsertLayerAbove', 'InsertLayerBelow', 'ChangeUpstreamLayer'])
        else:
            mutation_type = random.choice(['AdjustLayerParameter', 'InsertLayerAbove', 'InsertLayerBelow'])
            
        
        
        mutated_layer_id = random.choice(network.get_all_network_layer_ids())
        mutated_layer_type = network.get_network_layer_type(mutated_layer_id)
        
        print('Mutating network: %s. mutated_layer_id: %d (%s)' % (mutation_type, mutated_layer_id, mutated_layer_type))
        # Mutate one of the params.
        if mutation_type == 'AdjustLayerParameter':
            network.change_random_parameter_for_layer(mutated_layer_id)
        
        elif mutation_type == 'RemoveLayer':           
            network.delete_layer(mutated_layer_id)
        
        elif mutation_type == 'InsertLayerAbove':           
            network.insert_random_layer(network.get_upstream_layers(mutated_layer_id), [mutated_layer_id])
        
        elif mutation_type == 'InsertLayerBelow':           
            network.insert_random_layer([mutated_layer_id], network.get_downstream_layers(mutated_layer_id))
        
        elif mutation_type == 'ChangeUpstreamLayer':
            layer_options = [layer_id for layer_id in network.get_all_network_layer_ids() if layer_id != mutated_layer_id and layer_id not in network.get_upstream_layers(mutated_layer_id)]
            
            if len(layer_options) == 0:
                self.mutate(network)
            else:    
                new_upstream_layer_id = random.choice(layer_options)
                network.change_upstream_layer(mutated_layer_id, new_upstream_layer_id)
                
       
    
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
            
            shortest_network_layer_ids = shortest_network.get_all_network_layer_ids()
            longest_network_layer_ids = longest_network.get_all_network_layer_ids()
            
            for i in range(len(shortest_network_layer_ids)):
                current_shortest_network_layer_id = shortest_network_layer_ids[i]
                current_longest_network_layer_id = longest_network_layer_ids[i]
                chosen_layer_parameters = None
                chosen_layer_upstream_layers = None
                if random.random() > 0.5:
                    chosen_network = shortest_network
                    chosen_network_layer_id = current_shortest_network_layer_id
                else:
                    chosen_network = longest_network
                    chosen_network_layer_id = current_longest_network_layer_id
                                   
                chosen_layer_parameters, chosen_layer_upstream_layers = self.choose_parameters_and_upstream_connections_for_layer(chosen_network, chosen_network_layer_id, baby)                
                baby.add_layer_with_parameters(chosen_layer_parameters, chosen_layer_upstream_layers)
                    
                    
            for i in range(len(longest_network_layer_ids) - len(shortest_network_layer_ids)):
                
                if random.random() > 0.5:
                    chosen_layer_parameters, chosen_layer_upstream_layers = self.choose_parameters_and_upstream_connections_for_layer(longest_network, longest_network_layer_ids[i], baby)                
                    baby.add_layer_with_parameters(chosen_layer_parameters, chosen_layer_upstream_layers)
            
            
            babies.append(baby)

        return babies

    def choose_parameters_and_upstream_connections_for_layer(self, parent_network, layer_id, baby_network):
        chosen_layer_parameters = copy.deepcopy(parent_network.get_network_layer_details_dictionary(layer_id))
        chosen_layer_upstream_layers = parent_network.get_upstream_layers(layer_id)
        upstream_layers_which_exist_in_the_baby_network = [x for x in chosen_layer_upstream_layers if baby_network.has_a_layer_with_id(x)]
                
        if len(upstream_layers_which_exist_in_the_baby_network) == 0:
            network_ends = baby_network.get_network_layers_with_no_downstream_connections()
            if len(network_ends) == 0:
                chosen_layer_upstream_layers = None
            else:
                chosen_layer_upstream_layers = [random.choice(network_ends)]    
        
        return chosen_layer_parameters, chosen_layer_upstream_layers
                       
                
        
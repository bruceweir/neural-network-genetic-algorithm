"""Entry point to evolving the neural network. Start here."""
import argparse
import logging
from optimizer import Optimizer
from tqdm import tqdm
import time
import os
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot

parser = argparse.ArgumentParser(description='Generate neural networks via a Genetic Algorithm. Source: https://github.com/bruceweir/neural-network-genetic-algorithm. Originally forked from: https://github.com/harvitronix/neural-network-genetic-algorithm.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', 
                    help='The name of the dataset to use, either "mnist" or "cifar10"',
                    default='mnist')
parser.add_argument('-g', '--generations', 
                    help='The number of breeding generations to run for.',
                    type=int,
                    default=40)
parser.add_argument('-p', '--population_size', 
                    help='The size of the populations for each breeding cycle.',
                    type=int,
                    default=10)
parser.add_argument('-f', '--forbidden_layer_types',
                    help='An array of one or more layer types that should NOT be added to the networks. Options are Dense, Conv2D, MaxPooling2D',
                    nargs='+',
                    choices=['Dense', 'Conv2D', 'MaxPooling2D'],
                    default=[])
parser.add_argument('--mutate_chance',
                    help='The probability [0->1] that a particular network will undergo a random mutation during the breeding phase',
                    type=float,
                    default=0.2)
parser.add_argument('--random_select',
                    help='The probability [0->1] that a particular network which is not one of the best in a generation will not be culled',
                    type=float,
                    default=0.1)
parser.add_argument('--retain',
                    help='The proportion of best performing networks to keep to breed the next generation',
                    type=float,
                    default=0.4)
parser.add_argument('--initial_network_length',
                    help='The number of hidden layers that newly generated networks should have.',
                    type=int,
                    default=1)
parser.add_argument('--elitist',
                    help='Do not mutate the best candidate after every generation',
                    action='store_true')

args = parser.parse_args()

print(vars(args))
#parser.add_argument('dataset', 
#                    metavar='dataset', 
#                    type=string, 
#                    nargs=1, 
#                    help='The name of the dataset to use, either "mnist" or "cifar10"')

class Evolutionary_Neural_Network_Generator():
    """ Application class for creating neural networks via a genetic algorithm
    
    """
    
    
    def __init__(self, kwargs):
        
        self.save_directory = os.path.dirname(os.path.realpath(__file__))
        self.save_directory = os.path.join(self.save_directory, 'results')
        self.save_directory = os.path.join(self.save_directory, time.strftime("%c").replace(" ", "_").replace(":", "_"))
        os.makedirs(self.save_directory)
   
        # Setup logging.
        logging.basicConfig(
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
                level=logging.DEBUG,  
                filename= os.path.join(self.save_directory, 'log.txt')   
                )
        
        
        
        self.dataset = kwargs['dataset']
        self.generations = kwargs['generations']
        
        self.population_size = kwargs['population_size']
        self.initial_network_length = kwargs['initial_network_length']       
        
        self.optimizer = Optimizer(**kwargs)
        self.networks = self.optimizer.create_population(self.population_size, self.initial_network_length)        
        
        self.run_experiment()
    
    def train_networks(self, networks, dataset):
        """Train each network.
    
        Args:
            networks (list): Current population of networks
            dataset (str): Dataset to use for training/evaluating
        """
        pbar = tqdm(total=len(networks))
        for network in networks:
            network.train(dataset)
            pbar.update(1)
        pbar.close()

    def get_accuracy_stats(self, networks):
        """Get the average accuracy for a group of networks.
    
        Args:
            networks (list): List of networks
    
        Returns:
            float: The average accuracy of a population of networks.
    
        """
        total_accuracy = 0
        highest_accuracy = 0
        lowest_accuracy = 1
        highest_scoring_network = None
        
        for network in networks:
            total_accuracy += network.accuracy
            if network.accuracy > highest_accuracy:
                highest_accuracy = network.accuracy
                highest_scoring_network = network
            if network.accuracy < lowest_accuracy:
                lowest_accuracy = network.accuracy
    
        return total_accuracy / len(networks), highest_accuracy, lowest_accuracy, highest_scoring_network

    def run_evolutionary_generations(self):
        """Run the evolutionary algorithm over the number of generations sent to
        the constructor.
        """
        #optimizer = Optimizer(forbidden_layer_types)
        #networks = optimizer.create_population(population)
        
        
        # Evolve the generation.
        for i in range(self.generations):
            logging.info("***Doing generation %d of %d***" %
                         (i + 1, self.generations))
    
            self.train_networks(self.networks, self.dataset)
    
            average_accuracy, highest_accuracy, lowest_accuracy, highest_scoring_network = self.get_accuracy_stats(self.networks)       
    
            highest_scoring_network.save_network_details(os.path.join(self.save_directory, self.dataset + "_best_network_at_iteration_%d_acc%f" % (i, highest_accuracy)))
            
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
            logging.info("Generation best: %.2f%%" % (highest_accuracy * 100))
            logging.info("Generation worst: %.2f%%" % (lowest_accuracy * 100))
            logging.info('-'*80)
    
            # Evolve, except on the last iteration.
            if i != self.generations - 1:
                self.networks = self.optimizer.evolve(self.networks)
    
        self.networks = sorted(self.networks, key=lambda x: x.accuracy, reverse=True)
    
        self.print_networks(self.networks[:5])
    
        self.save_networks(self.dataset, self.networks[:5])
        
    #logging.shutdown()
    
    def print_networks(self, networks):
        """Print a list of networks.
    
        Args:
            networks (list): The population of networks
    
        """
        logging.info('-'*80)
        for network in networks:
            network.log_network()
            network.print_network_details()
                
    def save_networks(self, dataset, networks):
        
        """Save the trained models and a image of the networks.
    
        Args:
            dataset (string): The name of the dataset, which will preprended to the file name
            networks (list): The population of networks
            
    
        """
        for i in range(len(networks)):
            save_file_name = dataset + '-model_%d-' % i
            save_file_name = save_file_name + '_acc%.4f' % networks[0].accuracy
            save_file_name = os.path.join(self.save_directory, save_file_name)       
            networks[i].save_network_details(save_file_name)
      


    def run_experiment(self):
        """Evolve a network.
        
        dataset: The name of the data set to run on, currently either 'mnist' or 'cifar10'
        generations: The number of breeding generations to run over
        population: The breeding population at each step
        forbidden_layer_types:  An array of layer types that should NOT be used, options
                                currently are: 'Dense', 'Conv2D', 'MaxPooling2D'
        """
        
        logging.info("Running experiment with: %s" % vars(args))

        logging.info("***Evolving %d generations with population %d***" %
                     (self.generations, self.population_size))
    
        print('Saving results and log file to: ' + self.save_directory)    
        
        self.run_evolutionary_generations()


def draw_model_on_interactive_session(model):
        
    display(SVG(model_to_dot(model).create(prog='dot', format='svg')))

    
if __name__ == '__main__':
            
    Evolutionary_Neural_Network_Generator = Evolutionary_Neural_Network_Generator(vars(args))


#def make_and_draw(networks):
#    networks = optimizer.evolve(networks)
#    for n in networks:
#        model = compile_model(n, 10, (784,), (28, 28, 1))
#        display(SVG(model_to_dot(model).create(prog='dot', format='svg')))
#    return networks
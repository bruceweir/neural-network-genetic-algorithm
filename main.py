"""Entry point to evolving the neural network. Start here."""
import argparse
import logging
from optimizer import Optimizer
from tqdm import tqdm
import time
import os

parser = argparse.ArgumentParser(description='Generate neural networks via a Genetic Algorithm',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', 
                    help='The name of the dataset to use, either "mnist" or "cifar10."',
                    default='mnist')
parser.add_argument('-g', '--generations', 
                    help='The number of breeding generations to run for.',
                    type=int,
                    default=40)
parser.add_argument('-p', '--population', 
                    help='The size of the populations for each breeding cycle.',
                    type=int,
                    default=10)
parser.add_argument('-f', '--forbiddenlayers',
                    help='One or more layer types that should NOT be added to the networks. Options are Dense, Conv2D, MaxPooling2D',
                    nargs='+',
                    choices=['Dense', 'Conv2D', 'MaxPooling2D'])

args = parser.parse_args()
print('Dataset: ' + args.dataset)
print('Generations: %d' % args.generations)
print('Population: %d' % args.population)
if args.forbiddenlayers:
    print('Forbidden Layers: %s' % args.forbiddenlayers)


#parser.add_argument('dataset', 
#                    metavar='dataset', 
#                    type=string, 
#                    nargs=1, 
#                    help='The name of the dataset to use, either "mnist" or "cifar10"')

save_directory = os.path.dirname(os.path.realpath(__file__))
save_directory = os.path.join(save_directory, 'results')
save_directory = os.path.join(save_directory, time.strftime("%c").replace(" ", "_").replace(":", "_"))
os.makedirs(save_directory)
   
# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,  
    filename= os.path.join(save_directory, 'log.txt')
    
)


def train_networks(networks, dataset):
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

def get_accuracy_stats(networks):
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

def generate(generations, population, dataset, forbidden_layer_types):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        dataset (str): Dataset to use for training/evaluating
        forbidden_layer_types:  An array of layer types that should NOT be used, options
                            currently are: 'Dense', 'Conv2D', 'MaxPooling2D'

    """
    optimizer = Optimizer(forbidden_layer_types)
    networks = optimizer.create_population(population)
    
    
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        train_networks(networks, dataset)

        average_accuracy, highest_accuracy, lowest_accuracy, highest_scoring_network = get_accuracy_stats(networks)       

        highest_scoring_network.save_network_details(os.path.join(save_directory, dataset + "_best_network_at_iteration_%d_acc%f" % (i, highest_accuracy)))
        
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info("Generation best: %.2f%%" % (highest_accuracy * 100))
        logging.info("Generation worst: %.2f%%" % (lowest_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            networks = optimizer.evolve(networks)

    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    print_networks(networks[:5])

    save_networks(dataset, networks[:5])
    
    #logging.shutdown()

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.log_network()
        network.print_network_as_json()
        
def save_networks(dataset, networks):
    
    """Save the trained models and a image of the networks.

    Args:
        dataset (string): The name of the dataset, which will preprended to the file name
        networks (list): The population of networks
        

    """
    for i in range(len(networks)):
        save_file_name = dataset + '-model_%d-' % i
        save_file_name = save_file_name + '_acc%.4f' % networks[0].accuracy
        save_file_name = os.path.join(save_directory, save_file_name)       
        networks[i].save_network_details(save_file_name)
  


def run_experiment(dataset='mnist', generations=40, population=10, forbidden_layer_types=[]):
    """Evolve a network.
    
    dataset: The name of the data set to run on, currently either 'mnist' or 'cifar10'
    generations: The number of breeding generations to run over
    population: The breeding population at each step
    forbidden_layer_types:  An array of layer types that should NOT be used, options
                            currently are: 'Dense', 'Conv2D', 'MaxPooling2D'
    """
    
    
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    print('Saving results and log file to: ' + save_directory)    
    
    generate(generations, population, dataset, forbidden_layer_types)

if __name__ == '__main__':
 #   print(' ')
    run_experiment(args.dataset, args.generations, args.population, args.forbiddenlayers)
    

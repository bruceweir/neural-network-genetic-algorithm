"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import time

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
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

def generate(generations, population, dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer()
    networks = optimizer.create_population(population)
    
    
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        train_networks(networks, dataset)

        average_accuracy, highest_accuracy, lowest_accuracy, highest_scoring_network = get_accuracy_stats(networks)       

        highest_scoring_network.save_network_details(dataset + "_best_network_at_iteration_%d_acc%f" % (i, highest_accuracy))
        
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
        saveFileName = dataset + '-model_%d-' % i + time.strftime("%c").replace(" ", "_").replace(":", "_")
        saveFileName = saveFileName + '_acc%.4f' % networks[0].accuracy
    
        networks[i].save_network_details(saveFileName)
    
    
def main():
    """Evolve a network."""
    generations = 20  # Number of times to evolve the population.
    population = 10  # Number of networks in each generation.
    
    dataset = 'mnist' #'cifar10' or 'mnist'
 
   
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, dataset)

if __name__ == '__main__':
    main()
    
    

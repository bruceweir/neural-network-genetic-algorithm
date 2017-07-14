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

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

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

        average_accuracy = get_average_accuracy(networks)

        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            networks = optimizer.evolve(networks)

    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    print_networks(networks[:5])
    
    saveFileName = 'best_trained_model-' + time.strftime("%c").replace(" ", "_").replace(":", "_")
    saveFileName = saveFileName + '_acc%.4f' % networks[0].accuracy
    
    networks[0].save_model_image(saveFileName + ".png")
    networks[0].save_model(saveFileName + ".h5")
    
    

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.log_network()
        network.print_network_as_json()
        

    
def main():
    """Evolve a network."""
    generations = 2  # Number of times to evolve the population.
    population = 2  # Number of networks in each generation.
    
    dataset = 'mnist'
 
   
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, dataset)

if __name__ == '__main__':
    main()
    
    

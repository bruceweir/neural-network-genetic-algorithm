"""Entry point to evolving the neural network. Start here.

    Original project:  https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
    Extended by: bruce.weir@bbc.co.uk
    
    """
import argparse
import logging
from optimizer import Optimizer
from train import Train
from tqdm import tqdm
import time
import os
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import pickle


parser = argparse.ArgumentParser(description='Generate neural networks via a Genetic Algorithm. Source: https://github.com/bruceweir/neural-network-genetic-algorithm. Originally forked from: https://github.com/harvitronix/neural-network-genetic-algorithm.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', 
                    help='The name of a premade dataset to use for practise, either "mnist" or "cifar10"',
                    default=None)
parser.add_argument('--training_data',
                    help='The name of a saved numpy array containing the training data. The final column is the expected output values, the other columns are the input vector',
                    default=None)
parser.add_argument('--test_data',
                    help='The name of a saved numpy array containing the test data. The final column is the expected output values, the other columns are the input vector',
                    default=None)
parser.add_argument('--initial_network_population_file',
                    help='A pickled network population list to use as the starting population (this file gets output during a run, so by adding it back in you can resume your experiment). Note that the trained model is not included in this file, so any network is retrained.',
                    default=None)
parser.add_argument('--natural_input_shape',
                   help='For use when specifying your own training and test date files. The natural shape of the input data. For example, a 60x40x3 channel image would have a shape of "(60, 40, 3)"',
                   type=str)
parser.add_argument('-g', '--generations', 
                    help='The number of breeding generations to run for.',
                    type=int,
                    default=40)
parser.add_argument('-p', '--population_size', 
                    help='The size of the populations for each breeding cycle.',
                    type=int,
                    default=10)
parser.add_argument('-f', '--forbidden_layer_types',
                    help='One or more layer types that should NOT be added to the networks. Options are Dense, Conv2D, MaxPooling2D, ActivityRegularization, Dropout. Example: -f Conv2D MaxPooling2D',
                    nargs='+',
                    choices=['Dense', 'Conv2D', 'MaxPooling2D', 'ActivityRegularization', 'Dropout'],
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
parser.add_argument('--max_epochs', 
                    help='The trainer uses early stopping on the validation loss, but this setting will explicitly set the maximum number of training epochs to perform.',
                    type=int,
                    default=10000)
parser.add_argument('--batch_size',
                    help="If specifying your own training and test files, use this as the initial batch size for training.",
                    type=int,
                    default=64)
parser.add_argument('--is_classification',
                    help='Train a classification model (when using your own dataset. MNIST and CIFAR10 examples are always treated as classification problems)',
                    action='store_true',
                    default=False)

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
        
        
        
        self.generations = kwargs.get('generations', 20)
        self.dataset = kwargs.get('dataset', 'result')   
        if self.dataset is None:
            self.dataset = 'result'
        
        self.population_size = kwargs.get('population_size', 10)
        self.initial_network_length = kwargs.get('initial_network_length', 1)       
        
        self.optimizer = Optimizer(kwargs)
        
        if(kwargs['initial_network_population_file']) == None:
            self.networks = self.optimizer.create_population(self.population_size, self.initial_network_length)        
        else:
            with open(kwargs['initial_network_population_file'], 'rb') as network_population_file:
                self.networks = pickle.load(network_population_file)
                self.population_size = len(self.networks)
            
        
        #is_classification is dependent upon whether using mnist or cifar10 example data is being used
        #Train does the check. This bit could do with some refactoring
        self.train = Train(kwargs)
        self.is_classification = self.train.is_classification
        self.optimizer.is_classification = self.train.is_classification
        
        self.run_experiment()
    
    
    def run_experiment(self):
        """Evolve a network.
        """
        
        logging.info("Running experiment with: %s" % vars(args))

        logging.info("***Evolving %d generations with population %d***" %
                     (self.generations, self.population_size))
    
        print('Saving results and log file to: ' + self.save_directory)    
        
        self.run_evolutionary_generations()
    
    
    def run_evolutionary_generations(self):
        """Run the evolutionary algorithm over the number of generations sent to
        the constructor.
        """
        
        # Evolve the generation.
        for i in range(self.generations):
            logging.info("***Doing generation %d of %d***" %
                         (i + 1, self.generations))
    
            self.train_networks(self.networks)
    
            if self.is_classification:
                average_accuracy, highest_accuracy, lowest_accuracy, highest_scoring_network = self.get_accuracy_stats(self.networks)       
                
                highest_scoring_network.save_trained_model(os.path.join(self.save_directory, self.dataset + "_best_network_at_iteration_%d_acc%f" % (i, highest_accuracy)))
                
                logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
                logging.info("Generation best: %.2f%%" % (highest_accuracy * 100))
                logging.info("Generation worst: %.2f%%" % (lowest_accuracy * 100))
                logging.info('-'*80)
            else:
                average_loss, highest_loss, lowest_loss, best_scoring_network = self.get_loss_stats(self.networks)       
                
                best_scoring_network.save_trained_model(os.path.join(self.save_directory, self.dataset + "_best_network_at_iteration_%d_loss%f" % (i, lowest_loss)))
                
                logging.info("Generation average: %.2f%%" % (average_loss * 100))
                logging.info("Generation best: %.2f%%" % (highest_loss * 100))
                logging.info("Generation worst: %.2f%%" % (lowest_loss * 100))
                logging.info('-'*80)
            # Evolve, except on the last iteration.
            if i != self.generations - 1:
                self.networks = self.optimizer.evolve(self.networks)
            
            self.save_network_objects(self.networks)
    
        if self.is_classification:
            self.networks = sorted(self.networks, key=lambda x: x.accuracy, reverse=True)
        else:
            self.networks = sorted(self.networks, key=lambda x: x.loss, reverse=False)
    
        self.print_networks(self.networks[:5])
    
        self.save_trained_network_models(self.dataset, self.networks[:5])
        
    #logging.shutdown()     
        
    def train_networks(self, networks):
        """Train each network.
    
        Args:
            networks (list): Current population of networks
            
        """
        pbar = tqdm(total=len(networks))
        for network in networks:
            self.train.train_and_score(network)
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
        best_scoring_network = None
        
        for network in networks:
            total_accuracy += network.accuracy
            if network.accuracy > highest_accuracy:
                highest_accuracy = network.accuracy
                best_scoring_network = network
            if network.accuracy < lowest_accuracy:
                lowest_accuracy = network.accuracy
    
        return total_accuracy / len(networks), highest_accuracy, lowest_accuracy, best_scoring_network

    def get_loss_stats(self, networks):
        """Get the average accuracy for a group of networks.
    
        Args:
            networks (list): List of networks
    
        Returns:
            float: The average accuracy of a population of networks.
    
        """
        total_loss = 0
        highest_loss = 0
        lowest_loss = 1e99
        best_scoring_network = None
        
        for network in networks:
            total_loss += network.loss
            if network.loss < lowest_loss:
                lowest_loss = network.loss
                best_scoring_network = network
            if network.loss > highest_loss:
                highest_loss = network.loss
    
        return total_loss / len(networks), highest_loss, lowest_loss, best_scoring_network

    
    def print_networks(self, networks):
        """Print a list of networks.
    
        Args:
            networks (list): The population of networks
    
        """
        logging.info('-'*80)
        for network in networks:
            network.log_network()
            network.print_network_details()
                
    def save_trained_network_models(self, dataset, networks):
        
        """Save the trained models and a image of the networks.
    
        Args:
            dataset (string): The name of the dataset, which will preprended to the file name
            networks (list): The population of networks
            
    
        """
        for i in range(len(networks)):
            save_file_name = dataset + '-model_%d-' % i
            save_file_name = save_file_name + '_acc%.4f_loss%.4f' % (networks[i].accuracy, networks[i].loss)
            save_file_name = os.path.join(self.save_directory, save_file_name)       
            networks[i].save_trained_model(save_file_name)

    
    def save_network_objects(self, population_list):
        
        save_file_name = 'latest_network_population.pkl'
        save_file_name = os.path.join(self.save_directory, save_file_name) 
        
        with open(save_file_name, 'wb') as output:
            pickle.dump(population_list, output, pickle.HIGHEST_PROTOCOL)




def draw_model_on_interactive_session(model):
        
    display(SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')))

    
if __name__ == '__main__':
            
    Evolutionary_Neural_Network_Generator = Evolutionary_Neural_Network_Generator(vars(args))
 


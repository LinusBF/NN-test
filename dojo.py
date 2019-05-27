import sys

from backprop.manager import BackPropManager
from evolution.genotype import Genotype
from evolution.manager import EvolutionManager
from network.network import Network
from utils.training_data import get_training_batch_ten_arr
from utils.utils import save_genotype_data_to_file
import random as rand

TOPOLOGY = [10, 8, 4, 10]
NR_GENOTYPES = 500
BATCH_SIZE = 10
BATCH_FUNC = get_training_batch_ten_arr
FILE_PREFIX = "network_"


def evolution_based_training(topology, nr_genotypes, batch_size, batch_func, out_file_name):
    evo = EvolutionManager(topology)
    evo.start_evolution(nr_genotypes)

    best_genotype = None
    want_to_quit = False
    try:
        while not want_to_quit:
            evo.start_evaluation()
            for ins, out in batch_func(batch_size):
                evo.evaluate(ins, out)
            evo.evolve()
            print('Generation %d finished: %f\r' %
                  (evo.generations, evo.genotypes[0].fitness),
                  end="")
            best_genotype = evo.genotypes[0]

    except KeyboardInterrupt:
        want_to_quit = True

    print("\n")
    save_genotype_data_to_file(best_genotype, topology, out_file_name)
    print("Saved to file! Filename: " + out_file_name)

"""
name = "latest"
if len(sys.argv) > 1:
    name = sys.argv[1]

evolution_based_training(TOPOLOGY, NR_GENOTYPES, BATCH_SIZE, BATCH_FUNC, FILE_PREFIX + name)
"""

n = Network([2, 3, 3, 3, 2])
g = Genotype([0] * n.weight_count)
g.set_rand_params(-1, 1)
n.set_weights(g.params)

train_data = []
for i in range(10000):
    ins = [rand.random(), rand.random()]
    training_pair = [ins, ins]
    train_data.append(training_pair)

test_data = []
for i in range(1001):
    ins = [rand.random(), rand.random()]
    test_pair = [ins, ins]
    test_data.append(test_pair)

back_prop = BackPropManager(n)
back_prop.train_network(train_data, 10, test_data)
print("Done!")

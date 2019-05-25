import sys
from evolution.manager import EvolutionManager
from training_data import get_training_batch_ten_arr_rand, get_training_batch_ten_arr
from utils import save_genotype_data_to_file

TOPOLOGY = [10, 8, 4, 10]
NR_GENOTYPES = 500
BATCH_SIZE = 10
BATCH_FUNC = get_training_batch_ten_arr
FILE_PREFIX = "evo_10_arr_"


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


name = "training"
if len(sys.argv) > 1:
    name = sys.argv[1]

evolution_based_training(TOPOLOGY, NR_GENOTYPES, BATCH_SIZE, BATCH_FUNC, FILE_PREFIX + name)

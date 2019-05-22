import random
import csv
from evolution import EvolutionManager
from mnist import MNIST

mndata = MNIST("C:\Projects\python-mnist\data")
images, labels = mndata.load_training()

TOPOLOGY = [784, 16, 16, 10]
TRAINING_INDEX = 0


def save_genotype_to_file(genotype):
    with open('last_genotype.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        for p in genotype.params:
            writer.writerow([str(p)])

    out_file.close()


def get_inputs_and_answer():
    global TRAINING_INDEX
    img = images[TRAINING_INDEX]
    wanted_output = []
    for i in range(10):
        wanted_output.append(1 if i is labels[TRAINING_INDEX] else 0)
    TRAINING_INDEX = TRAINING_INDEX + 1
    return [img, wanted_output]


def evolution_based_training(topology, nr_of_genotypes, nr_of_generations):
    evo_manager = EvolutionManager(topology)
    evo_manager.start_evolution(nr_of_genotypes)
    while evo_manager.generations < nr_of_generations:
        evo_manager.start_evaluation()
        inputs, wanted_output = get_inputs_and_answer()
        evo_manager.evolve(inputs, wanted_output)

        if evo_manager.generations % 50 is 0:
            parent1 = evo_manager.genotypes[0]
            print("Generation " + str(evo_manager.generations) + ":")
            print("Achieved a maximum fitness of: " + "{:.4f}".format(parent1.fitness))
            print("Wanted output was: " + ", ".join(str(x) for x in wanted_output))
            print("Best Genotype: " + ", ".join("{:.2f}".format(x) for x in evo_manager.get_sorted_agents()[0].output))


def get_rand_input():
    nr = random.randint(0, 9)
    inputs = []
    for i in range(10):
        inputs.append(1 if i is nr else 0)
    output = [1, 0] if nr > 5 else [0, 1]
    return [inputs, inputs]


def get_ten_inputs():
    in_out_pairs = []
    for nr in range(10):
        inputs = []
        for i in range(10):
            inputs.append(1 if i is nr else 0)
        output = [1, 0] if nr > 5 else [0, 1]
        in_out_pairs.append([inputs, inputs])
    return in_out_pairs


def get_input():
    nr = int(input("Pick a number between 0-9\n"))
    inputs = []
    for i in range(10):
        inputs.append(1 if i is nr else 0)
    output = [1, 0] if nr > 5 else [0, 1]
    return [inputs, inputs]


# evolution_based_training(TOPOLOGY, 50, 100000)

top = [10, 8, 8, 10]
evo = EvolutionManager(top)
evo.start_evolution(100)

best_genotype = None
want_to_quit = False

try:
    while not want_to_quit:
        evo.start_evaluation()
        for ins, out in get_ten_inputs():
            evo.evaluate(ins, out)
        evo.evolve()
        print("Gen: " + str(evo.generations))
        print(str(evo.genotypes[0].fitness))
        best_genotype = evo.genotypes[0]

except KeyboardInterrupt:
    want_to_quit = True


save_genotype_to_file(best_genotype)

while True:
    ins, out = get_input()
    evo.evaluate(ins, out)
    print(str(evo.genotypes[0].fitness))
    for i, o in enumerate(evo.get_sorted_agents()[0].network.outputs):
        print("Layer " + str(i + 1) + ": " + ", ".join("{:.2f}".format(x) for x in o))

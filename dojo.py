import random

from evolution import EvolutionManager, Genotype, GenerationManager, Agent
from mnist import MNIST
from os import path

from network import Layer, Network

mndata = MNIST("C:\Projects\python-mnist\data")
images, labels = mndata.load_training()

TOPOLOGY = [784, 16, 16, 10]
TRAINING_INDEX = 0


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


def get_input():
    nr = int(input("Pick a number between 0-9\n"))
    inputs = []
    for i in range(10):
        inputs.append(1 if i is nr else 0)
    output = [1, 0] if nr > 5 else [0, 1]
    return [inputs, inputs]


# evolution_based_training(TOPOLOGY, 50, 100000)

top = [10, 16, 16, 10]
evo = EvolutionManager(top)
evo.start_evolution(4)

while evo.generations < 1000:
    evo.start_evaluation()
    ins, out = get_rand_input()
    evo.evolve(ins, out)
    print(str(evo.genotypes[0].fitness))

while True:
    evo.start_evaluation()
    ins, out = get_input()
    evo.evolve(ins, out)
    print(str(evo.genotypes[0].fitness))
    print("Best agent: " + ", ".join("{:.2f}".format(x) for x in evo.get_sorted_agents()[0].output))

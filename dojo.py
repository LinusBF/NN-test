import random
import csv
from evolution import EvolutionManager, Genotype
from mnist import MNIST

from network import Network

mndata = MNIST("C:\Projects\python-mnist\data")
images, labels = mndata.load_training()

TOPOLOGY = [784, 16, 16, 10]
TRAINING_INDEX = 0


def save_genotype_to_file(genotype):
    with open('last_genotype.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        for p in genotype.params:
            writer.writerow([str(p)])

    out_file.close()


def get_genotype_from_file(filename):
    weights = []
    with open(filename, 'r', newline='\n') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            weights.append(float(row[0]))

    in_file.close()
    return weights


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

"""
net = Network([10, 4, 4, 10])
genotype = Genotype([0] * net.weight_count)
genotype.set_rand_params(-1, 1)
net.set_weights(genotype.params)
print(net.layers[0].weights)
print(net.layers[1].weights)
print(net.layers[2].weights)
"""



top = [10, 10, 10]
evo = EvolutionManager(top)
evo.start_evolution(500)

best_genotype = None
want_to_quit = False

try:
    while not want_to_quit:
        evo.start_evaluation()
        for ins, out in get_ten_inputs():
            evo.evaluate(ins, out)
        evo.evolve()
        print('Generation %d finished: %f\r' % (evo.generations, evo.genotypes[0].fitness), end="")
        #sorted_agents = evo.get_sorted_agents()
        best_genotype = evo.genotypes[0]

        #if evo.generations % 100 is 0:
        #    print("\n")
        #    print(evo.get_sorted_agents()[0].network)

except KeyboardInterrupt:
    want_to_quit = True

print("\n")
save_genotype_to_file(best_genotype)
print("Saved to file!")


"""
w = get_genotype_from_file("last_genotype_perfect_1.csv")
net = Network([10, 10])
net.set_weights(w)

while True:
    ins, out = get_input()
    ins.append(1)
    print("Answer: " + ", ".join("{:.2f}".format(x) for x in net.process_input(ins)))
    print(net)
"""

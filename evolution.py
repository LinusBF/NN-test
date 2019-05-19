import random

from network import Network


class EvolutionManager:
    def __init__(self, topology):
        self.generations = 0
        self.topology = topology
        self.genotypes = []
        self.agents = []

    def start_evolution(self, nr_of_genotypes, generation_ready_callback):
        self.genotypes = []
        temp_network = Network(self.topology)
        for i in range(nr_of_genotypes):
            new_genotype = Genotype([0] * temp_network.weight_count)
            new_genotype.set_rand_params(-1, 1)
            self.genotypes.append(new_genotype)
        self.start_evaluation()
        generation_ready_callback()

    def start_evaluation(self):
        agents = []
        for g in self.genotypes:
            agents.append(Agent(g, self.topology))
        self.agents = agents

    def evolve(self, inputs, correct):
        for a in self.agents:
            a.evaluate(inputs, correct)
        new_genotypes = GenerationManager.get_better_population(self.genotypes)
        self.generations = self.generations + 1
        self.genotypes = new_genotypes


class GenerationManager:
    @staticmethod
    def get_better_population(c_pop: list):
        eval_sum = 0
        for genotype in c_pop:
            eval_sum = eval_sum + genotype.eval
        for genotype in c_pop:
            genotype.set_fitness(eval_sum)

        c_pop.sort()
        int_pop = GenerationManager.select(c_pop)
        new_pop = GenerationManager.recombine(int_pop, len(c_pop))

        for p in new_pop:
            p.mutate()

        return new_pop

    @staticmethod
    def select(pop):
        return pop.copy()[0:3]

    @staticmethod
    def recombine(pop, size):
        new_pop = [pop[0], pop[1]]
        while len(new_pop) < size:
            parent1 = pop[random.randint(0, len(pop) - 1)]
            parent2 = pop[random.randint(0, len(pop) - 1)]
            [child1, child2] = GenerationManager.create_children(parent1, parent2)
            new_pop.append(child1)
            if len(new_pop) < size:
                new_pop.append(child2)

        return new_pop

    @staticmethod
    def create_children(parent1, parent2):
        off_param1 = []
        off_param2 = []

        for i in range(len(parent1.params)):
            if random.random() < 0.6:  # Swap params
                off_param1[i] = parent2.params[i]
                off_param2[i] = parent1.params[i]
            else:  # Don't swap
                off_param1[i] = parent1.params[i]
                off_param2[i] = parent2.params[i]

        return [Genotype(off_param1), Genotype(off_param2)]


class Agent:
    def __init__(self, genotype, topology):
        self.alive = True
        self.genotype = genotype
        self.network = Network(topology)
        current_param = 0
        for layer in self.network.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] = genotype.params[current_param]
                    current_param = current_param + 1

    def evaluate(self, inputs, correct):
        network_answer = self.network.process_input(inputs)
        diff_sum = 0
        for i in range(len(network_answer)):
            diff_sum = diff_sum + correct[i] - network_answer[i]
        evaluation = diff_sum/len(network_answer)
        self.genotype.eval = evaluation
        return evaluation


class Genotype:
    def __init__(self, params):
        self.params = params
        self.eval = 0
        self.fitness = 0

    def __cmp__(self, other):
        return self.fitness > other.fitness

    def set_rand_params(self, min_val, max_val):
        range_val = max_val - min_val
        for i in range(len(self.params)):
            self.params[i] = random.random() * range_val + min_val

    def set_fitness(self, bench):
        self.fitness = self.eval / bench

    def mutate(self):
        for i in range(self.params):
            if random.random() < 0.3:
                if random.random() > 0.5:
                    self.params[i] = self.params[i] * (random.random() * 2)
                else:
                    self.params[i] = self.params[i] * (random.random() * -2)

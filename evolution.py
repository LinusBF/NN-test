import random

from network import Network


class EvolutionManager:
    def __init__(self, topology):
        self.generations = 0
        self.topology = topology
        self.genotypes = []
        self.agents = []
        self.avg_fitness = 0

    def start_evolution(self, nr_of_genotypes):
        self.genotypes = []
        temp_network = Network(self.topology)
        for i in range(nr_of_genotypes):
            new_genotype = Genotype([0] * temp_network.weight_count)
            new_genotype.set_rand_params(-1, 1)
            self.genotypes.append(new_genotype)
        self.start_evaluation()

    def start_evaluation(self):
        agents = []
        for g in self.genotypes:
            agents.append(Agent(g, self.topology))
        self.agents = agents

    def evolve(self, inputs, correct):
        for a in self.agents:
            a.evaluate(inputs, correct)
        new_genotypes, avg_fit = GenerationManager.get_better_population(self.genotypes)
        self.avg_fitness = avg_fit
        mutated_genotypes = EvolutionManager.mutate(new_genotypes)
        self.generations = self.generations + 1
        self.genotypes = mutated_genotypes

    def get_sorted_agents(self):
        return sorted(self.agents, reverse=True)

    @staticmethod
    def mutate(pop):
        mutated_pop = [pop[0], pop[1]]  # Don't mutate best two genotypes
        for genotype in pop[2:]:
            genotype.mutate()
            mutated_pop.append(genotype)
        return mutated_pop


class GenerationManager:
    @staticmethod
    def get_better_population(c_pop: list):
        eval_sum = 0
        for genotype in c_pop:
            eval_sum = eval_sum + genotype.eval

        fitness_sum = 0
        for genotype in c_pop:
            genotype.set_fitness(eval_sum/len(c_pop))
            fitness_sum = fitness_sum + genotype.fitness

        c_pop.sort(reverse=True)
        int_pop = GenerationManager.get_int_pop(c_pop)
        new_pop = GenerationManager.recombine(int_pop, len(c_pop))

        return [new_pop, eval_sum]

    @staticmethod
    def get_int_pop(pop):
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
        off_param1 = [0] * len(parent1.params)
        off_param2 = [0] * len(parent1.params)

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
        self.genotype = genotype
        self.output = []
        self.network = Network(topology)
        current_param = 0
        for layer in self.network.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] = genotype.params[current_param]
                    current_param = current_param + 1

    def __lt__(self, other):
        return self.genotype.fitness < other.genotype.fitness

    def evaluate(self, inputs, correct):
        network_answer = self.network.process_input(inputs)
        self.output = network_answer
        evaluation = Agent.eval_progress(network_answer, correct)
        self.genotype.eval = evaluation
        return evaluation

    @staticmethod
    def eval_diff(answer, correct):
        diff_sum = 0
        for i in range(len(answer)):
            diff_sum = diff_sum + correct[i] - answer[i]
        return diff_sum / len(answer)

    @staticmethod
    def eval_nr_incorrect(answer, correct):
        incorrect = 0
        for i in range(len(answer)):
            diff = correct[i] - answer[i]
            if abs(diff) > 0.01:
                incorrect = incorrect + 1
        return incorrect

    @staticmethod
    def eval_progress(answer, correct):
        sum_percent = 0
        for i in range(len(answer)):
            if correct[i] is 1:
                sum_percent = sum_percent + answer[i]
            else:
                sum_percent = sum_percent + (1 - answer[i])
        return sum_percent / len(answer)


class Genotype:
    def __init__(self, params):
        self.params = params
        self.eval = 0
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def set_rand_params(self, min_val, max_val):
        range_val = max_val - min_val
        for i in range(len(self.params)):
            self.params[i] = random.random() * range_val + min_val

    def set_fitness(self, bench):
        self.fitness = self.eval  # / bench

    def mutate(self):
        for i in range(len(self.params)):
            if random.random() < 0.5:
                if random.random() > 0.5:
                    self.params[i] = self.params[i] * (random.random() * 2)
                else:
                    self.params[i] = self.params[i] * (random.random() * -2)

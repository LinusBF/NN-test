import random

from evolution.genotype import Genotype


class GenerationManager:
    @staticmethod
    def get_better_population(c_pop: list):
        for genotype in c_pop:
            genotype.set_fitness()

        c_pop.sort()
        int_pop = GenerationManager.get_int_pop(c_pop)
        new_pop = GenerationManager.recombine(int_pop, len(c_pop))

        for genotype in new_pop[-5:]:
            genotype.set_rand_params(-1, 1)

        return new_pop

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

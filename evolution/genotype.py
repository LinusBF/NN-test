import random


class Genotype:
    def __init__(self, params):
        self.params = params
        self.eval = []
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def set_rand_params(self, min_val, max_val):
        range_val = max_val - min_val
        for i in range(len(self.params)):
            self.params[i] = random.random() * range_val + min_val

    def set_fitness(self):
        eval_sum = 0
        for e in self.eval:
            eval_sum = eval_sum + e
        self.fitness = eval_sum / len(self.eval)

    def mutate(self):
        for i in range(len(self.params)):
            if random.random() < 0.3:
                if random.random() > 0.5:
                    self.params[i] = self.params[i] * (random.random() * 2)
                else:
                    self.params[i] = self.params[i] * (random.random() * -2)

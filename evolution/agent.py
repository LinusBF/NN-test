import math

from network import Network


class Agent:
    def __init__(self, genotype, topology):
        self.genotype = genotype
        self.genotype.eval = []
        self.output = []
        self.network = Network(topology)
        self.network.set_weights(genotype.params)

    def __lt__(self, other):
        return self.genotype.fitness < other.genotype.fitness

    def evaluate(self, inputs, correct):
        network_answer = self.network.process_input(inputs)
        self.output = network_answer
        evaluation = Agent.eval_diff(network_answer, correct)
        self.genotype.eval.append(evaluation)
        return evaluation

    @staticmethod
    def eval_diff(answer, correct):
        diff_sum = 0
        for i in range(len(answer)):
            diff_sum = diff_sum + math.pow(answer[i] - correct[i], 2)
        return diff_sum

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

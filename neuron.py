import random
import math


class Neuron:
    def __init__(self, size):
        self.weights = []
        for x in range(size):
            pos_or_neg = random.random() > 0.5
            w = random.random() if pos_or_neg else (random.random() * -1)
            self.weights.append(w)
        self.bias = random.random()
        self.inputs = []

    def feed(self, inputs):
        self.inputs = inputs

    def calc(self):
        value = 0
        for idx, w in enumerate(self.weights):
            value = value + (w * self.inputs[idx])
        return self.sig(value + self.bias)

    def change_weights(self, weights):
        self.weights = weights

    @staticmethod
    def sig(value):
        return math.exp(value) / (math.exp(value) + 1.0)


if __name__ == "__main__":
    i = []
    for x in range(784):
        i.append(random.random())
    print(i)
    n = Neuron(784)
    n.feed(i)
    print(n.calc())

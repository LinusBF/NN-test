import math


class Layer:
    def __init__(self, neurons: int, output_neurons: int):
        self.neurons = neurons
        self.output_neurons = output_neurons
        self.weights = [[0 for j in range(output_neurons)] for k in range(neurons + 1)]  # +1 for bias weight

    def process_input(self, inputs):
        if len(inputs) > self.neurons:
            raise ValueError("Amount of inputs has to match amount of input neurons in layer!")

        sums = [0] * self.output_neurons
        for j in range(self.output_neurons):  # For each output neuron
            for k in range(self.neurons):  # For input neuron
                sums[j] += inputs[k] * self.weights[k][j]
            sums[j] += self.weights[self.neurons][j]  # Add Bias

        # Apply activation function to keep layer output within the range 0.0 -> 1.0
        for s in range(len(sums)):
            sums[s] = self.activate_function(sums[s])

        return sums

    @staticmethod
    def activate_function(value):
        return Layer.sigmoid(value)

    @staticmethod
    def sigmoid(value):
        """
        Simple Sigmoid function to determine the activation of the neuron
        :param value: float
        :return: float
        """
        if value > 100:
            return 1
        elif value < -100:
            return 0
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def soft_sigmoid(value):
        """
        The SoftSign function as proposed by Xavier Glorot and Yoshua Bengio (2010):
        "Understanding the difficulty of training deep feedforward neural networks".
        :param value: float
        :return: float
        """
        return abs(value / (abs(value) + 1.0))

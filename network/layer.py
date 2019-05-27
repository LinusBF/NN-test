import math
import numpy as np


class Layer:
    def __init__(self, neurons: int, output_neurons: int):
        self.neurons = neurons
        self.output_neurons = output_neurons
        self.weights = [[0 for j in range(output_neurons)] for k in range(neurons + 1)]  # +1 for bias weight

    def process_input(self, inputs):
        if inputs.size > self.neurons:
            print(inputs.size, self.neurons)
            raise ValueError("Amount of inputs has to match amount of input neurons in layer!")

        neuron_values = np.zeros((self.output_neurons, 1))
        sums = np.zeros((self.output_neurons, 1))
        for j in range(self.output_neurons):  # For each output neuron
            for k in range(self.neurons):  # For input neuron
                sums[j][0] += inputs[k][0] * self.weights[k][j]
            sums[j][0] += self.weights[self.neurons][j]  # Add Bias
            neuron_values[j][0] = sums[j][0]

        # Apply activation function to keep layer output within the range 0.0 -> 1.0
        for s in range(sums.size):
            sums[s][0] = self.activate_function(sums[s][0])

        return [sums, neuron_values]

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
        return 1.0 / (1.0 + np.exp(-value))

    @staticmethod
    def soft_sigmoid(value):
        """
        The SoftSign function as proposed by Xavier Glorot and Yoshua Bengio (2010):
        "Understanding the difficulty of training deep feedforward neural networks".
        :param value: float
        :return: float
        """
        return abs(value / (abs(value) + 1.0))

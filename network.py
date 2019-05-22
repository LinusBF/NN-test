import math


class Network:
    def __init__(self, topology):
        self.topology = topology

        # Calc total amount of weights of all neurons in the network
        total_weights = 0
        for l in range(len(topology) - 1):
            total_weights = total_weights + ((topology[l] + 1) * topology[l + 1])  # +1 for bias
        self.weight_count = total_weights

        # Setup all the layers in the network
        self.layers = []
        for l in range(len(topology) - 1):
            self.layers.append(Layer(topology[l], topology[l + 1]))  # +1 for bias

    def process_input(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.process_input(output)
        return output


class Layer:
    def __init__(self, neurons, output_neurons):
        self.neurons = neurons
        self.output_neurons = output_neurons
        self.weights = [[0] * output_neurons] * (neurons + 1)

    def process_input(self, inputs):
        sums = [0] * self.output_neurons
        biased_input = inputs.copy()
        biased_input.append(1.0)  # Bias neuron

        for j in range(len(self.weights[0])):  # For each output neuron
            for i in range(len(self.weights)):  # For each weight in input neuron (+ bias neuron)
                sums[j] = sums[j] + biased_input[i] * self.weights[i][j]

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
        return (1.0 / (1.0 + math.exp(-1 * value))) if value > -100.0 else 1.0

    @staticmethod
    def soft_sigmoid(value):
        """
        The SoftSign function as proposed by Xavier Glorot and Yoshua Bengio (2010):
        "Understanding the difficulty of training deep feedforward neural networks".
        :param value: int
        :return: int
        """
        return abs(value / (abs(value) + 1.0))

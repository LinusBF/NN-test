import math


class Network:
    def __init__(self, topology):
        self.topology = topology
        self.outputs = []

        # Calc total amount of weights of all neurons in the network
        total_weights = 0
        for l in range(len(topology) - 1):
            total_weights = total_weights + ((topology[l] + 1) * topology[l + 1])  # +1 for bias
        self.weight_count = total_weights

        # Setup all the layers in the network
        self.layers = []
        for l in range(len(topology) - 1):
            self.layers.append(Layer(topology[l], topology[l + 1]))  # +1 for bias

    def set_weights(self, weights):
        current_weight = 0
        for layer in self.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] = weights[current_weight]
                    current_weight += 1

    def process_input(self, inputs):
        self.outputs = []
        output = inputs
        for layer in self.layers:
            output.append(1)
            output = layer.process_input(output)
            self.outputs.append(output)
        return output

    def __str__(self):
        s = "Network:\n"
        for idx, layer in enumerate(self.layers):
            s += "\tLayer " + str(idx) + ": "
            s += str(layer.weights) + "\n"
        for idx, out in enumerate(self.outputs):
            s += "\tOutput " + str(idx) + ": "
            s += str(out) + "\n"
        return s


class Layer:
    def __init__(self, neurons, output_neurons):
        self.neurons = neurons
        self.output_neurons = output_neurons
        self.weights = [[0 for i in range(output_neurons)] for j in range(neurons + 1)]

    def process_input(self, inputs):
        sums = [0 for i in range(self.output_neurons)]
        for j in range(self.output_neurons):  # For each output neuron
            for k in range(self.neurons):  # For input neuron
                sums[j] += inputs[k] * self.weights[k][j]
            sums[j] += self.weights[self.neurons][j]  # Add Bias

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
        if (value > 100):
            return 1
        elif (value < -100):
            return 0
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def soft_sigmoid(value):
        """
        The SoftSign function as proposed by Xavier Glorot and Yoshua Bengio (2010):
        "Understanding the difficulty of training deep feedforward neural networks".
        :param value: int
        :return: int
        """
        return abs(value / (abs(value) + 1.0))

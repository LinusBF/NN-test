from network.layer import Layer


class Network:
    def __init__(self, topology):
        """

        :type topology: [int] An list of integers, representing the number of neurons in each layer
        """
        self.topology = topology
        self.activations = []
        self.neuron_values = []

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
        """
        Applies the weights given to all layers in the network
        :type weights: float[] A list of floats that correspond to the weight between every pair of neurons in the
        network on the format:
        If the network has the topology [2, 3, 2] =>
        weight[0] is the weight from the 1st input neuron in the 1st layer to the first output neuron of that layer
        weight[4] is the weight from the 2nd input neuron in the 1st layer and the 2nd output neuron of that layer
        weight[7] is the weight from the 1st input neuron in the 2nd layer and the 2nd output neuron of that layer
        """
        current_weight = 0
        for layer in self.layers:
            for i in range(len(layer.weights)):  # Each input neuron in the layer
                for j in range(len(layer.weights[i])):  # Each output neuron in the layer
                    layer.weights[i][j] = weights[current_weight]
                    current_weight += 1

    def process_input(self, inputs):
        self.activations = []
        self.neuron_values = []
        activation = inputs
        for layer in self.layers:
            activation, neuron_values = layer.process_input(activation)
            self.activations.append(activation)
            self.neuron_values.append(neuron_values)
        return activation

    def __str__(self):
        s = "Network: (" + str(self.topology) + ")\n"
        for idx, layer in enumerate(self.layers):
            s += "\tLayer " + str(idx) + ":\n"
            for w in layer.weights:
                s += "\t\t["
                s += ", ".join("{:.2f}".format(x) for x in w[:10])
                s += ", ...]\n" if len(w) > 9 else "]\n"
            s += "\n"

        for idx, out in enumerate(self.activations):
            s += "\tOutput " + str(idx) + ": "
            s += ", ".join("{:.2f}".format(x) for x in out) + "\n"
        return s

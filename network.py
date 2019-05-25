from layer import Layer


class Network:
    def __init__(self, topology):
        """

        :type topology: [int] An list of integers, representing the number of neurons in each layer
        """
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
        self.outputs = []
        output = inputs
        for layer in self.layers:
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

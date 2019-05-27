import numpy as np

from network.layer import Layer


class Batch:
    def __init__(self, network):
        self.network = network
        self.weights = []
        self.biases = []
        self.weight_adj = []
        self.bias_adj = []
        self.get_weights_and_biases_from_net(network)

    def get_weights_and_biases_from_net(self, network):
        self.weights = []
        self.biases = []
        for idx, layer in enumerate(network.layers):
            self.weights.append(np.zeros((layer.output_neurons, layer.neurons)))
            self.biases.append(np.zeros((layer.output_neurons, 1)))
            for j in range(layer.output_neurons):  # For each output neuron
                for k in range(layer.neurons):  # For input neuron
                    self.weights[idx][j, k] = layer.weights[k][j]
                self.biases[idx][j, 0] = layer.weights[layer.neurons][j]

    def run_batch(self, batch_inputs):
        for training_pair in batch_inputs:
            train_input = np.array([training_pair[0]]).transpose()
            train_correct = np.array([training_pair[1]]).transpose()
            self.process_training_data(train_input, train_correct)
        return self.compute_batch_result()

    def process_training_data(self, inputs, correct):
        self.network.process_input(inputs)
        nabla_w, nabla_b = self.backprop(correct)
        self.weight_adj.append(nabla_w)
        self.bias_adj.append(nabla_b)

    def backprop(self, correct):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        activations = self.network.activations
        neuron_values = self.network.neuron_values

        deltas = (activations[-1] - correct) * self.sigmoid_prime(neuron_values[-1])
        nabla_b[-1] = deltas
        nabla_w[-1] = np.dot(deltas, activations[-2].transpose())

        for layer_index in range(2, len(self.network.layers)):
            neuron_value = neuron_values[layer_index]
            prime_activations = self.sigmoid_prime(neuron_value)
            deltas = np.dot(self.weights[-layer_index+1].transpose(), deltas) * prime_activations
            nabla_b[-layer_index] = deltas
            nabla_w[-layer_index] = np.dot(deltas, activations[-layer_index-1].transpose())

        return nabla_w, nabla_b

    def compute_batch_result(self):
        computed_w = [np.zeros(nabla_w_layer.shape) for nabla_w_layer in self.weight_adj[0]]
        for nabla_w in self.weight_adj:
            for idx, layer in enumerate(nabla_w):
                computed_w[idx] += layer
        for layer in computed_w:
            layer /= len(self.weight_adj)

        computed_b = [np.zeros(nabla_b_layer.shape) for nabla_b_layer in self.bias_adj[0]]
        for nabla_b in self.bias_adj:
            for idx, layer in enumerate(nabla_b):
                computed_b[idx] += layer
        for layer in computed_b:
            layer /= len(self.bias_adj)

        return [computed_w, computed_b]

    @staticmethod
    def sigmoid_prime(value):
        """
        Simple Sigmoid function to determine the activation of the neuron
        :param value: float
        :return: float
        """
        return Layer.sigmoid(value) * (1 - Layer.sigmoid(value))


if __name__ == "__main__":
    from network.network import Network
    from evolution.genotype import Genotype
    net = Network([2, 3, 2])
    gen = Genotype([0] * net.weight_count)
    gen.set_rand_params(-1, 1)
    net.set_weights(gen.params)
    batch = Batch(net)

    ins = [[0.25, 0.65], [0.35, 0.75], [2.3, 5.3]]
    training_data = [ins, ins]
    computed_w, computed_b = batch.run_batch(training_data)

    print("test")

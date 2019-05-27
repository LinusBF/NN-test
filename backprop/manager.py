import numpy as np

from backprop.batch import Batch


class BackPropManager:
    def __init__(self, network):
        self.network = network

    def train_network(self, training_data, batch_size, testing_data):
        batch_index = 0
        while batch_index < len(training_data):
            batch = Batch(self.network)
            batch_to_run = training_data[batch_index: batch_index + batch_size]
            weight_gradient, bias_gradient = batch.run_batch(batch_to_run)
            self.improve_network(weight_gradient, bias_gradient)
            batch_index += batch_size
            print("Batch " + str(batch_index/batch_size) + " completed!")
            test = np.array([testing_data[int(batch_index/batch_size)][0]]).transpose()
            output = self.network.process_input(test)
            print("Network answered the input\n", test, "\nwith\n", output)

        return True

    def improve_network(self, w_grad, b_grad):
        for idx, layer in enumerate(self.network.layers):
            for j in range(layer.output_neurons):  # For each output neuron
                for k in range(layer.neurons):  # For input neuron
                    layer.weights[k][j] -= w_grad[idx][j, k]
                layer.weights[layer.neurons][j] -= b_grad[idx][j][0]

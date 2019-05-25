import sys

from network import Network
from utils import get_genotype_data_from_file


def get_test_ten_arr():
    nr = int(input("Pick a number between 0-9\n"))
    inputs = []
    for i in range(10):
        inputs.append(1 if i is nr else 0)
    return [inputs, inputs]


def ten_arr_test(file_name):
    weights, topology = get_genotype_data_from_file(file_name + ".csv")
    net = Network(topology)
    net.set_weights(weights)

    while True:
        ins, out = get_test_ten_arr()
        print("Answer: " + ", ".join("{:.2f}".format(x) for x in net.process_input(ins)))
        print(net)


name = "network_latest"
if len(sys.argv) > 1:
    name = sys.argv[1]

ten_arr_test(name)

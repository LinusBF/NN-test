import random


def get_training_batch_ten_arr_rand(batch_size):
    in_out_pairs = []
    nr = 0
    for t in range(batch_size):
        inputs = []
        for i in range(10):
            inputs.append(1 if i is nr else 0)
        in_out_pairs.append([inputs, inputs])
        nr = nr + 1 if nr < 9 else 0
    random.shuffle(in_out_pairs)
    return in_out_pairs


def get_training_batch_ten_arr(batch_size):
    in_out_pairs = []
    nr = 0
    for t in range(batch_size):
        inputs = []
        for i in range(10):
            inputs.append(1 if i is nr else 0)
        in_out_pairs.append([inputs, inputs])
        nr = nr + 1 if nr < 9 else 0
    return in_out_pairs

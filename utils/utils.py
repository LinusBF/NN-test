import csv


def save_genotype_data_to_file(genotype, topology, fn):
    with open(fn + '.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(str(x) for x in topology)
        for p in genotype.params:
            writer.writerow([str(p)])

    out_file.close()


def get_genotype_data_from_file(filename):
    weights = []
    with open(filename, 'r', newline='') as in_file:
        reader = csv.reader(in_file)
        topology = next(reader, None)
        for row in reader:
            weights.append(float(row[0]))

    in_file.close()
    return [weights, [int(x) for x in topology]]

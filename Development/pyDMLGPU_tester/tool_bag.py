from locals import *
from pyDMLGPU import Apriori, FastItemMiner


def pick_algorithm(algorithm_type):
    if algorithm_type == PARALLEL_APRIORI_ALGORITHM:
        print('pick_tool: ', PARALLEL_APRIORI_ALGORITHM)
        return Apriori()
    elif algorithm_type == PARALLEL_MINER_ALGORITHM:
        print('pick_tool: ', PARALLEL_MINER_ALGORITHM)
        return FastItemMiner()
    else:
        raise TypeError('Algorithm type is not selected!')


def get_names_file(path):
    names_dictionary = dict()
    with open(path, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            name, code = process_names_line(line)
            names_dictionary[int(code)] = name
    print(names_dictionary)
    return names_dictionary


def process_names_line(line):
    line = line.strip()
    line = line.split(' ')
    code = line[-1]
    del line[-1]
    return ' '.join(line), code

import numpy as np
import itertools


def get_database(file_name):
    with open(file_name, 'r') as data_file:
        data = data_file.readlines()
        result = [process_string_line_to_list(x) for x in data]
        return result


def process_string_line_to_list(line):
    # print(str(line))
    # line = line.replace(' \n', '')
    line = line.strip()
    line = line.split(' ')
    line = list(map(int, line))
    return line


def process_data_base(data):
    # contains lengths of each line
    v_length = list()
    # contains starting index of each line
    v_start_index = list()
    last = 0
    for i in data:
        v_length.append(len(i))
        v_start_index.append(last)
        last += len(i)

    v_length = np.array(v_length, dtype=np.int32)
    v_start_index = np.array(v_start_index, dtype=np.int32)

    merged_data_base = list(itertools.chain.from_iterable(data))
    merged_data_base = np.array(merged_data_base, dtype=np.int32)

    # save number of transactions
    number_transactions = len(data)

    return merged_data_base, v_start_index, v_length, number_transactions


# if __name__ == '__main__':
#     data = get_database()
#     for i in data:
#         print(i)

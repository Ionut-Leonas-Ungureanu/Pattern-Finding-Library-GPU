import numpy as np
import pyopencl as cl
import itertools


def get_database(file_name):
    with open(file_name, 'r') as data_file:
        data = data_file.readlines()
        result = [process_string_line_to_list(x) for x in data]
        return result


def process_string_line_to_list(line):
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


def make_groups_of_transactions(context, number_transactions, work_group_size):
    # make array's for group transactions
    transactions_per_group = number_transactions // work_group_size
    left_transactions = number_transactions % work_group_size

    group_transactions_length = [transactions_per_group + 1 if i < left_transactions else transactions_per_group
                                 for i in range(work_group_size)]

    group_transactions_length = np.array(group_transactions_length, dtype=np.int32)
    buffer_group_transactions_length = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                                                 hostbuf=group_transactions_length)
    stamp = 0
    group_transactions_start = [0] * work_group_size
    for i in range(work_group_size):
        group_transactions_start[i] = stamp
        stamp += group_transactions_length[i]
    group_transactions_start = np.array(group_transactions_start, dtype=np.int32)
    buffer_group_transactions_start = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR,
                                                hostbuf=group_transactions_start)
    return buffer_group_transactions_length, buffer_group_transactions_start


def remove_from_database(database, elements, db_items):
    elements = set([element for item_set in elements for element in item_set])
    if elements != db_items:
        db = [[item for item in transaction if item in elements] for transaction in database]
        return [sublist for sublist in db if sublist != []]
    else:
        return database


# if __name__ == '__main__':
#     data = get_database()
#     for i in data:
#         print(i)

import pyopencl as cl
import numpy as np
import itertools
import math
from pyDMLGPU.py_apriori.kernels import APRIORI_2D_KERNEL


def process_item_sets_on_gpu(candidates_item_sets, database, v_start_index, v_lengths, k_step,
                             nr_transactions, nr_transactions_database, min_supp, gpu_setter, threshold=np.inf):
    item_sets = candidates_item_sets

    # get number of item sets
    number_item_sets = len(item_sets)

    context = gpu_setter.get_context()

    program = cl.Program(context, APRIORI_2D_KERNEL).build().process_item_sets
    queue = gpu_setter.make_queue()

    mem_flags = cl.mem_flags
    buffer_database = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=database)
    buffer_lengths = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v_lengths)
    buffer_start_index = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v_start_index)

    # get max size of work group
    work_group_size = gpu_setter.get_device().max_work_group_size
    nr_items = gpu_setter.get_device().max_work_item_sizes
    nr_items = nr_items[0]
    # make array's for group transactions
    transactions_per_group = nr_transactions // work_group_size
    left_transactions = nr_transactions % work_group_size

    group_transactions_length = [transactions_per_group+1 if i < left_transactions else transactions_per_group for i in
                                 range(work_group_size)]

    group_transactions_length = np.array(group_transactions_length, dtype=np.int32)
    buffer_group_transactions_length = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                                                 hostbuf=group_transactions_length)
    stamp = 0
    group_transactions_start = [0] * work_group_size
    for i in range(work_group_size):
        group_transactions_start[i] = stamp
        stamp += group_transactions_length[i]
    group_transactions_start = np.array(group_transactions_start, dtype=np.int32)
    buffer_group_transactions_start = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                                                hostbuf=group_transactions_start)

    # check if number of item-sets is bigger than the available number of work items
    if number_item_sets > nr_items*nr_items:
        threshold = nr_items*nr_items

    if number_item_sets > threshold:
        number_chunks = number_item_sets/threshold
        number_chunks = math.ceil(number_chunks)

        for j in range(int(number_chunks)):
            try:
                chunk_item_sets = item_sets[(j*threshold): (j*threshold+threshold)]
                v_frequency_log = np.zeros(int(len(chunk_item_sets)), dtype=np.int)
            except IndexError:
                chunk_item_sets = item_sets[(j*threshold):]
                v_frequency_log = np.zeros(len(chunk_item_sets), dtype=np.int)

            if k_step > 1:
                chunk_item_sets_merged = list(itertools.chain.from_iterable(chunk_item_sets))
            else:
                chunk_item_sets_merged = chunk_item_sets

            chunk_item_sets_merged = np.array(chunk_item_sets_merged, dtype=np.int)

            buffer_item_sets = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                                         hostbuf=chunk_item_sets_merged)
            buffer_frequency_log = cl.Buffer(context, mem_flags.COPY_HOST_PTR, hostbuf=v_frequency_log)

            program(queue, (work_group_size, len(chunk_item_sets)), (work_group_size, 1), buffer_item_sets,
                    buffer_database, buffer_group_transactions_length, buffer_group_transactions_start,
                    np.int32(k_step), np.int32(nr_transactions_database), np.float32(min_supp), buffer_start_index,
                    buffer_lengths, buffer_frequency_log, cl.LocalMemory(4*work_group_size))

            cl.enqueue_copy(queue, v_frequency_log, buffer_frequency_log)

            return [chunk_item_sets[i] for i in range(v_frequency_log.size) if v_frequency_log[i] == 1]
    else:
        # item sets buffer is not bigger than threshold
        v_frequency_log = np.zeros(len(item_sets), dtype=np.int)

        buffer_item_sets = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                                     hostbuf=np.array(item_sets, dtype=np.int))
        buffer_frequency_log = cl.Buffer(context, mem_flags.COPY_HOST_PTR, hostbuf=v_frequency_log)

        program(queue, (work_group_size, len(item_sets)), (work_group_size, 1), buffer_item_sets, buffer_database,
                buffer_group_transactions_length, buffer_group_transactions_start, np.int32(k_step),
                np.int32(nr_transactions_database), np.float32(min_supp), buffer_start_index, buffer_lengths,
                buffer_frequency_log, cl.LocalMemory(4*work_group_size))

        cl.enqueue_copy(queue, v_frequency_log, buffer_frequency_log)

        return [item_sets[i] for i in range(v_frequency_log.size) if v_frequency_log[i] == 1]


# if __name__ == '__main__':
#     test_dictionary = {}
#     l = list()
#     l.append([1, 3, 4])
#     l.append([2, 3, 5])
#     l.append([1, 2, 3, 5])
#     l.append([2, 5])
#
#     v_length = list()
#     v_start_index = list()
#     last = 0
#     for i in l:
#         v_length.append(len(i))
#         v_start_index.append(last)
#         last += len(i)
#
#     v_length = np.array(v_length)
#     v_start_index = np.array(v_start_index)
#
#     itemset = np.array([2, 4])
#
#     database1D = [item for sublist in l for item in sublist]
#     database1D = np.array(database1D)
#
#     check_itemset_on_gpu(test_dictionary, database1D, v_start_index, v_length, itemset, len(l), 0.5)
#
#     print(test_dictionary)

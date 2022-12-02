import pyopencl as cl
import numpy as np
import math
from pyDMLGPU.py_miner.kernels import GET_FREQUENT_ITEMS_KERNEL, FREQUENT_ITEM_SET_MINER
from pyDMLGPU.py_miner.container import Container
from pyDMLGPU.py_miner.utils import buffer_from_1d, buffer_from_2d, buffer_from_3d, make_array_from_3d
from pyDMLGPU.helper import make_groups_of_transactions
from timeit import default_timer as timer


def get_frequent_items(context, items_array, buffer_database, buffer_start_index, buffer_lengths, nr_transactions,
                       min_supp, work_group_size, nr_gpu_items, threshold=np.inf):
    frequent_items = list()

    buffer_group_transactions_length, \
        buffer_group_transactions_start = make_groups_of_transactions(context, nr_transactions,
                                                                      work_group_size)

    program = cl.Program(context, GET_FREQUENT_ITEMS_KERNEL).build().get_frequents
    queue = cl.CommandQueue(context)

    mem_flags = cl.mem_flags

    if len(items_array) > nr_gpu_items*nr_gpu_items:
        threshold = nr_gpu_items*nr_gpu_items

    # if GPU processing takes to long , the desktop will freeze
    if threshold < len(items_array):
        number_chunks = len(items_array) / threshold
        number_chunks = math.ceil(number_chunks)

        for j in range(int(number_chunks)):
            try:
                chunk_item_sets = items_array[(j * threshold): (j * threshold + threshold)]

            except IndexError:
                chunk_item_sets = items_array[(j * threshold):]

            v_frequents = np.zeros(int(len(chunk_item_sets)), dtype=np.int)

            buffer_items = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                                     hostbuf=np.array(chunk_item_sets, dtype=np.int))
            buffer_results = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v_frequents)

            program(queue, (work_group_size, len(chunk_item_sets)), (work_group_size, 1), buffer_items, buffer_database,
                    buffer_group_transactions_length, buffer_group_transactions_start, buffer_start_index,
                    buffer_lengths, np.int32(nr_transactions), np.float32(min_supp), buffer_results,
                    cl.LocalMemory(work_group_size*4))

            cl.enqueue_copy(queue, v_frequents, buffer_results)

            frequent_items = [items_array[i] for i in range(v_frequents.size) if v_frequents[i] == 1]

    else:
        buffer_items = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=np.array(items_array,
                                                                                                          dtype=np.int))
        v_frequents = np.zeros(len(items_array), dtype=np.int)
        buffer_results = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=v_frequents)

        program(queue, (work_group_size, len(items_array)), (work_group_size, 1), buffer_items, buffer_database,
                buffer_group_transactions_length, buffer_group_transactions_start, buffer_start_index, buffer_lengths,
                np.int32(nr_transactions), np.float32(min_supp), buffer_results, cl.LocalMemory(work_group_size*4))

        cl.enqueue_copy(queue, v_frequents, buffer_results)
        frequent_items = [items_array[i] for i in range(v_frequents.size) if v_frequents[i] == 1]

    return frequent_items


def launch_miner(context, items, buffer_database, buffer_database_start_index, buffer_database_lengths,
                 nr_transactions_database, nr_transactions_modified, min_supp, work_group_size, nr_gpu_items):
    results = dict()

    buffer_group_transactions_length, \
        buffer_group_transactions_start = make_groups_of_transactions(context, nr_transactions_modified,
                                                                      work_group_size)

    # make items container
    items_container = Container.load_data(items, nr_gpu_items)

    read_container = items_container
    data = read_container.values[0]
    load_item_set_to_dict(results, data)

    # make expansion

    steps = len(items_container.heights)
    max_height = max(items_container.heights)

    # prepare program for GPU execution
    queue = cl.CommandQueue(context)

    program = cl.Program(context, FREQUENT_ITEM_SET_MINER).build().miner

    for i in range(steps - 1):
        write_container = read_container.generate_write_container(items_container.heights, items_container.widths)

        # make buffers
        # items_container buffers
        buffer_items_value = buffer_from_3d(context, items_container.values)
        buffer_items_heights = buffer_from_1d(context, items_container.heights)
        buffer_items_start_items = buffer_from_2d(context, items_container.start_of_item_sets)
        buffer_items_widths = buffer_from_2d(context, items_container.widths)
        buffer_items_start_2d = buffer_from_1d(context, items_container.start_2d_parsing_buffer)

        number_nodes = len(read_container.heights) - 1

        # read_container buffers
        buffer_read_values = buffer_from_3d(context, read_container.values)
        # print('read_values', read_container.values)
        buffer_read_heights = buffer_from_1d(context, read_container.heights)
        # print('heights', read_container.heights)
        buffer_read_widths = buffer_from_2d(context, read_container.widths)
        # print('widths', read_container.widths)
        buffer_read_start_index = buffer_from_1d(context, read_container.end_values_per_node)
        # print('read_start_index', read_container.start_values_per_node)
        buffer_read_start_2d = buffer_from_1d(context, read_container.start_2d_parsing_buffer)
        # print('read_start_2d', read_container.start_2d_parsing_buffer)

        # write_container buffers
        buffer_write_values = buffer_from_3d(context, write_container.values)
        # print('write values', write_container.values)
        write_like_array = make_array_from_3d(write_container.values)
        buffer_write_start_index = buffer_from_2d(context, write_container.start_of_item_sets)
        # print('write start item_sets', write_container.start_of_item_sets)
        buffer_write_start_2d = buffer_from_1d(context, write_container.start_2d_parsing_buffer)

        # launch opencl program
        program(queue, (max_height, number_nodes, work_group_size), (1, 1, work_group_size),
                buffer_database, buffer_group_transactions_length, buffer_group_transactions_start,
                buffer_database_start_index, buffer_database_lengths,
                buffer_items_value, buffer_items_heights, buffer_items_widths,
                buffer_items_start_items, buffer_items_start_2d,
                buffer_write_values, buffer_write_start_index, buffer_write_start_2d,
                buffer_read_values, buffer_read_heights, buffer_read_widths,
                buffer_read_start_index, buffer_read_start_2d,
                np.int32(nr_transactions_database), np.int32(number_nodes), np.float32(min_supp),
                cl.LocalMemory(4*work_group_size))

        # retrieve data
        cl.enqueue_copy(queue, write_like_array, buffer_write_values)

        # load data into write_container and clean
        write_container.fill_from_1d_synonymous(write_like_array)

        # save data from write_container.values[0] and delete it so you won't carry unwanted memory
        data = write_container.values[0]
        load_item_set_to_dict(results, data)

        items_container.remove_last_node()
        read_container = write_container

    # get frequent items with the biggest length
    list_keys = list(results.keys())
    max_key = max(list_keys)
    for key in list_keys:
        if key < max_key:
            del results[key]

    return results[max_key]


def load_item_set_to_dict(results: dict, data):
    for item_set in data:
        key = len(item_set)
        if key not in results:
            results[key] = list()
        results[key].append(item_set)


# if __name__ == '__main__':
#     items = [32, 38, 39, 41, 48]
#     launch_miner(items, None, None, None, None, None)

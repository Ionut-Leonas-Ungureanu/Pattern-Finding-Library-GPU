from pyDMLGPU.py_miner.gpu_feeder import get_frequent_items, launch_miner
from pyDMLGPU.helper import process_data_base, remove_from_database
import pyopencl as cl
from timeit import default_timer as timer


def run_gpu_fast_itemset_miner(database, min_supp, gpu_setter):
    v_database, v_start_index, v_lengths, number_transactions_database = process_data_base(database)

    # distinct items contained by database
    lot = set(v_database)

    # get items in database
    items = list(lot)
    items.sort()
    items = [[item] for item in items]

    context = gpu_setter.get_context()

    # database buffers
    buffer_database = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_database)
    buffer_database_start_index = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                            hostbuf=v_start_index)
    buffer_database_lengths = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_lengths)

    # get work group size
    work_group_size = gpu_setter.get_device().max_work_group_size
    # get number of work items
    nr_gpu_items = gpu_setter.get_device().max_work_item_sizes[0]

    # get frequent items using GPU
    items = get_frequent_items(context, items, buffer_database, buffer_database_start_index, buffer_database_lengths,
                               number_transactions_database, min_supp, work_group_size, nr_gpu_items)

    # clean database
    database = remove_from_database(database, items, lot)
    v_database, v_start_index, v_lengths, number_transactions_modified = process_data_base(database)

    # database buffers
    buffer_database = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_database)
    buffer_database_start_index = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                            hostbuf=v_start_index)
    buffer_database_lengths = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_lengths)

    # run miner using GPU
    answer = launch_miner(context, items, buffer_database, buffer_database_start_index, buffer_database_lengths,
                          number_transactions_database, number_transactions_modified, min_supp, work_group_size,
                          nr_gpu_items)

    return answer

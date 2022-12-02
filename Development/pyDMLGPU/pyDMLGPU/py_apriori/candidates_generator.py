import numpy as np
import pyopencl as cl
from pyDMLGPU.py_apriori.kernels import APRIORI_CANDIDATES_GENERATION


def generate_candidates_gpu(item_sets, k, gpu_setter):
    array_container = None
    total_future_nr_item_sets = 0

    # divide the item_sets list in chunks of length equal with max number of work items
    number_work_items = gpu_setter.get_device().max_work_item_sizes[0]

    number_full_chunks = int(len(item_sets)/number_work_items)
    chunks_list = list()

    if number_full_chunks > 0:
        chunks_list = [item_sets[i*number_work_items:i*number_work_items + number_work_items]
                       for i in range(number_full_chunks)]

    # get the rest of item_sets
    chunks_list.append(item_sets[number_full_chunks*number_work_items:])

    # process each chunk
    for item_set in chunks_list:
        nr_item_sets = len(item_set)
        flat_item_sets = np.hstack(item_set)

        n = nr_item_sets - 1
        future_nr_item_sets = int((n * n + n) / 2)
        total_future_nr_item_sets += future_nr_item_sets

        generated_candidates = np.zeros(future_nr_item_sets * (k + 1), dtype=np.int32)

        value = 0
        starts = list()
        starts.append(value)

        for i in range(0, nr_item_sets - 2):
            value += ((nr_item_sets - 1 - i) * (k + 1))
            starts.append(value)

        starts_generated_candidates = np.array([starts], dtype=np.int32)

        context = gpu_setter.get_context()

        buffer_item_sets = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=flat_item_sets)
        buffer_generated_candidates = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                                hostbuf=generated_candidates)
        buffer_starts = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=starts_generated_candidates)

        queue = gpu_setter.make_queue()
        program = cl.Program(context, APRIORI_CANDIDATES_GENERATION).build().generate_candidates

        program(queue, (nr_item_sets, nr_item_sets - 1, 1), (1, 1, 1), buffer_item_sets, np.int32(nr_item_sets),
                np.int32(k),
                buffer_generated_candidates, buffer_starts, cl.LocalMemory(k * 4), cl.LocalMemory(k * 4),
                cl.LocalMemory((k + 1) * 4))

        cl.enqueue_copy(queue, generated_candidates, buffer_generated_candidates)

        if array_container is not None:
            array_container = np.concatenate((array_container, generated_candidates), axis=None)
        else:
            array_container = generated_candidates

    # flatten result_list
    array_container = array_container.tolist()
    results_list = set([tuple(array_container[i * (k + 1):i * (k + 1) + (k + 1)])
                        for i in range(total_future_nr_item_sets)])
    results_list = [list(t) for t in results_list if all(x == 0 for x in list(t)) is not True]

    return results_list


# if __name__ == '__main__':
    # a = np.array([0, 1])
    # b = np.array([0, 2])
    # c = np.array([1, 2])
    # d = np.array([1, 3])
    #
    # my_dict = dict()
    # my_dict[hash(tuple(a))] = a
    # my_dict[hash(tuple(b))] = b
    # my_dict[hash(tuple(c))] = c
    # my_dict[hash(tuple(d))] = d

    # my_dict = generate_candidates(my_dict)
    # print(my_dict)

    # item_sets = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    # k = 2
    # gpu_setter = GPUSetter()
    # print(generate_candidates_gpu(item_sets, k, gpu_setter))


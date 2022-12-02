import copy
from pyDMLGPU.py_apriori.candidates_generator import generate_candidates_gpu
from pyDMLGPU.py_apriori.gpu_feeder import process_item_sets_on_gpu
from pyDMLGPU.helper import process_data_base, remove_from_database
from timeit import default_timer as timer


def run_gpu_apriori(database, min_supp, gpu_setter):
    answer = list()
    v_database, v_start_index, v_lengths, number_transactions_database = process_data_base(database)
    number_transactions_modifiable = number_transactions_database

    # distinct items contained by database
    lot = set(v_database)

    k = 1
    # while condition
    go_to_next = True
    frequent_item_sets = []

    while go_to_next is True:
        if k == 1:
            candidates_item_sets = [[i] for i in lot]
        else:
            candidates_item_sets = generate_candidates_gpu(frequent_item_sets, k-1, gpu_setter)

        if k == 2:
            database = remove_from_database(database, frequent_item_sets, lot)
            v_database, v_start_index, v_lengths, number_transactions_modifiable = process_data_base(database)

        # exit if there are no candidates
        if len(candidates_item_sets) == 0:
            break

        frequent_item_sets = process_item_sets_on_gpu(candidates_item_sets, v_database,
                                                      v_start_index, v_lengths, k, number_transactions_modifiable,
                                                      number_transactions_database, min_supp, gpu_setter)

        # test stop condition
        if len(frequent_item_sets) <= 1:
            go_to_next = False
        else:
            go_to_next = True
            k += 1

        if len(frequent_item_sets) != 0:
            answer = copy.deepcopy(list(frequent_item_sets))

    return answer


# if __name__ == '__main__':
#     # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#     l = list()
#     l.append([1, 3, 4])
#     l.append([2, 3, 5])
#     l.append([1, 2, 3, 5])
#     l.append([2, 5])
#
#     database, v_start_index, v_length, nr_t = process_data_base(l)
#
#     my_apriori_obj = Apriori()
#     start = timer()
#     my_apriori_obj.run(database, v_start_index, v_length, nr_t, 0.5)
#     end = timer()
#     print("Total: ", end-start)
#     print(my_apriori_obj.answer)

from generate_candidates import generate_candidates
import copy
import itertools
from timeit import default_timer as timer
from helper import remove_from_database


class AprioriSeq:

    def __init__(self):
        self._answer = list()

    def run(self, database, min_supp):
        number_transactions = len(database)
        lot = set(list(itertools.chain.from_iterable(database)))
        k = 1
        go_to_next = True
        frequent_item_sets = list()
        candidates_list = list()
        while go_to_next is True:
            if k == 1:
                candidates_list = [[i] for i in lot]
            else:
                candidates_list = generate_candidates(frequent_item_sets, k)

            if k == 2:
                database = remove_from_database(database, frequent_item_sets, lot)

            frequent_item_sets = list()

            for item_set in candidates_list:
                count = 0
                for transaction in database:
                    if all(x in transaction for x in item_set):
                        count += 1

                supp = count / number_transactions

                if supp >= min_supp:
                    frequent_item_sets.append(item_set)

            # print("frequent items: ", frequent_item_sets)

            if len(frequent_item_sets) <= 1:
                go_to_next = False
            else:
                go_to_next = True
                k += 1

            if len(frequent_item_sets) != 0:
                self._answer = copy.deepcopy(frequent_item_sets)

    def get_results(self):
        return self._answer

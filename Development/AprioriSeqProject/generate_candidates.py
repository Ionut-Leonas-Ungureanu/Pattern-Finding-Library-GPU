

def generate_candidates(frequent_item_sets, k):
    aux_dictionary = dict()

    for i in range(len(frequent_item_sets)-1):
        first = frequent_item_sets[i]
        for second in frequent_item_sets[i+1:]:
            candidate = set(first) | set(second)
            if len(candidate) == k:
                candidate = list(candidate)
                candidate.sort()
                aux_dictionary[hash(tuple(candidate))] = candidate

    return aux_dictionary.values()

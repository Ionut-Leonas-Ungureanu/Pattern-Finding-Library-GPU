from timeit import default_timer as timer
from helper import process_data_base, get_database
from apriori import AprioriSeq


if __name__ == '__main__':
    apriori_seq_obj = AprioriSeq()

    database_list = ['retail.dat', 'T40I10D100K.dat', 'chess.dat', 'mushroom.dat']
    supports = [[0.022], [0.036], [0.82], [0.4]]

    for database_name in database_list:
        database = get_database(database_name)
        idx = database_list.index(database_name)
        for supp in supports[idx]:
            start_m = timer()
            apriori_seq_obj.run(database, supp)
            end_m = timer()
            print("{0} : results {1} in {2} seconds with length {3}"
                  .format(database_name, apriori_seq_obj.get_results(), end_m - start_m,
                          len(apriori_seq_obj.get_results())))

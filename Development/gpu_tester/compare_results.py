from pyDMLGPU import Apriori, FastItemMiner
from helper import process_data_base, get_database
from timeit import default_timer as timer


def log(string, mode):
    f = open('log.txt', mode)
    f.write(string)
    f.close()


def warm_up():
    print('\t Warm up started!')
    for i in range(15):
        warm_alg = Apriori()
        warm_alg.set_database(get_database('retail.dat'))
        warm_alg.set_minimum_support(0.8)
        warm_alg.run()
    print('\t Warm up ended!')


if __name__ == '__main__':
    test_database = ['retail.dat', 'T40I10D100K.dat', 'chess.dat', 'mushroom.dat']
    test_supports = [[0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 10],
                     [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6],
                     [70, 72, 74, 76, 78, 80, 82],
                     [28, 30, 32, 34, 36, 38, 40]]

    test_algorithms = [Apriori(), FastItemMiner()]
    test_alg_names = ['Apriori', 'FIMiner']

    log_string = 'Logs: \n'
    log(log_string, 'w')

    warm_up()

    for idx_database in range(len(test_database)):
        print('Start Testing {0}'.format(test_database[idx_database]))
        database = get_database(test_database[idx_database])

        for alg in test_algorithms:
            alg.set_database(database)

        log_string = '*' * 100 + ' ' + test_database[idx_database] + ' ' + '*' * 100 + '\n'
        log(log_string, 'a')

        for supp_value in test_supports[idx_database]:
            print('\tTesting for supp {0}'.format(supp_value))

            log_string = '%' * 100 + ' ' + test_database[idx_database] + ' ' + '%' * 100 + '\n'
            log(log_string, 'a')

            times_list = list()
            # launch
            for alg in test_algorithms:
                alg.set_minimum_support(supp_value)
                start = timer()
                alg.run()
                end = timer()
                print(test_alg_names[test_algorithms.index(alg)], end-start)
                times_list.append(end - start)

            # get results
            test_results = list()

            for alg in test_algorithms:
                test_results.append(alg.get_results())

            flag = True

            # test lengths
            length_apriori = len(test_results[0])
            length_miner = len(test_results[1])

            if length_apriori != length_miner:
                flag = False

            if flag is True:
                for item_set in test_results[0]:
                    if item_set not in test_results[1]:
                        flag = False

            if flag is True:
                log_string = 'TEST FOR SUPPORT ~ {0}% ~ OK\n'.format(supp_value)
                log_string += '\tResults for {0} have length {1}\n'.format(test_alg_names[0], length_apriori)
                log_string += '\tResults for {0} have length {1}\n'.format(test_alg_names[1], length_miner)
            else:
                log_string = '>>>>>>> Test for {0} FAILED \n'.format(supp_value)

            for idx in range(len(test_results)):
                log_string += '\t{0}:: {1} seconds for {2}\n'.format(test_alg_names[idx],
                                                                     times_list[idx],
                                                                     test_results[idx])
            log(log_string, 'a')

            print('\tFinished tests for support {0}'.format(supp_value))

    print('Finished ...')

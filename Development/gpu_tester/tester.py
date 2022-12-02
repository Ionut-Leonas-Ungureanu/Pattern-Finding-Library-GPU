
from timeit import default_timer as timer
from helper import process_data_base, get_database
from pyDMLGPU import Apriori, FastItemMiner

if __name__ == '__main__':
    # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    apriori_obj = Apriori()
    miner = FastItemMiner()
    supp = 2.2

    # database, v_start_index, v_length, nr_t = process_data_base(get_database('webdocs.dat'))
    # print("database: ", database)
    # print("start: ", v_start_index)
    # print(len(v_start_index))
    # print("lengths: ", v_length)
    # print(len(v_length))
    # print("tranzactii: ", nr_t)
    # print("<< Data base processed >>")

    # start_m = timer()
    # miner.run(database, v_start_index, v_length, nr_t, 0.1)
    # end_m = timer()
    # print("tester:: miner: result {0} in {1} seconds with length {2}".format(end_m - start_m, miner.get_results(),
    #                                                                          len(miner.get_results())))

    database = get_database('retail.dat')
    print('apriori len db', len(database))
    apriori_obj.set_database(database)
    print("<< Starting parallel Apriori >>")

    start_time = timer()
    apriori_obj.set_minimum_support(50)
    apriori_obj.run()
    end_time = timer()

    print("tester:: apriori: result = {0} in {1} seconds with length {2}".format(apriori_obj.get_results(),
                                                                                 end_time - start_time,
                                                                                 len(apriori_obj.get_results())))

    print('miner len db', len(database))
    miner.set_database(database)
    miner.set_minimum_support(50)
    start_m = timer()
    miner.run()
    end_m = timer()
    print("tester:: miner: result {0} in {1} seconds with length {2}".format(miner.get_results(), end_m - start_m,
                                                                             len(miner.get_results())))









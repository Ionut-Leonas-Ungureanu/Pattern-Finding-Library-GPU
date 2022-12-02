
from timeit import default_timer as timer
from pyDMLGPU.helper import process_data_base, get_database
from pyDMLGPU.py_apriori.apriori import run_gpu_apriori
from pyDMLGPU.gpu_settings_singleton import GPUSetter


if __name__ == '__main__':
    # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    # apriori_obj = Apriori()
    # miner = FastItemMiner()
    supp = 0.8

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
    # apriori_obj.set_database(database)
    print("<< Starting parallel Apriori >>")

    for i in range(30):
        start_time = timer()
        # apriori_obj.set_minimum_support(0.8)
        #     #     # apriori_obj.run()
        results = run_gpu_apriori(database, 0.1, GPUSetter())
        end_time = timer()

        print("tester:: apriori: result = {0} in {1} seconds with length {2}".format(results,
                                                                                     end_time - start_time,
                                                                                     len(results)))

    # print('miner len db', len(database))
    # miner.set_database(database)
    # miner.set_minimum_support(0.8)
    # start_m = timer()
    # miner.run()
    # end_m = timer()
    # print("tester:: miner: result {0} in {1} seconds with length {2}".format(end_m - start_m, miner.get_results(),
    #                                                                          len(miner.get_results())))





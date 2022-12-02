from abc import ABCMeta, abstractmethod
from pyDMLGPU.helper import process_data_base, get_database
from pyDMLGPU.py_apriori.apriori import run_gpu_apriori
from pyDMLGPU.py_miner.fast_itemset_miner import run_gpu_fast_itemset_miner
from pyDMLGPU.gpu_settings_singleton import GPUSetter


class DataMiningAlgorithm(metaclass=ABCMeta):
    """
    Superclass for data mining algorithms which provides required interface and methods.
    """

    def __init__(self):
        self._database = None

        self._min_supp = 0

        self._gpu_setter = GPUSetter()

        self._results = None

        self._able_to_run = False

    @abstractmethod
    def set_database(self, new_database):
        self.__check_database(new_database)

    @abstractmethod
    def set_minimum_support(self, new_min_sup):
        self.__check_minimum_support(new_min_sup)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_results(self):
        pass

    def __check_database(self, database):
        if not isinstance(database, list):
            self._able_to_run = False
            raise TypeError('database must be of type list.')

        for transaction in database:
            if not isinstance(transaction, list):
                self._able_to_run = False
                raise TypeError('transactions in database must be of type list.')

            if not all([isinstance(item, int) for item in transaction]):
                self._able_to_run = False
                raise TypeError('Database must have only integer values.')

            # sort transactions in database ascending
            transaction.sort()

        self._able_to_run = True
        self._database = database

    def __check_minimum_support(self, min_sup):

        try:
            min_sup = float(min_sup)
        except ValueError:
            self._able_to_run = False
            raise ValueError('min_supp value must be of type float.')

        if min_sup >= 100 or min_sup <= 0:
            self._able_to_run = False
            raise ValueError('min_supp value must be between 0% and 100%, but not 0% or 100%.')

        self._min_supp = float(min_sup / 100)
        self._able_to_run = True


class Apriori(DataMiningAlgorithm):
    """
    Implementation for Apriori algorithm.
    """

    def __init__(self):
        super().__init__()

    def set_database(self, new_database):
        super().set_database(new_database)

    def set_minimum_support(self, new_min_sup):
        super().set_minimum_support(new_min_sup)

    def run(self):
        if self._able_to_run is True:
            self._results = run_gpu_apriori(self._database, self._min_supp, self._gpu_setter)
        else:
            print('can not run')

    def get_results(self):
        return self._results


class FastItemMiner(DataMiningAlgorithm):
    """
    Implementation for Fast Item Miner algorithm.
    """

    def __init__(self):
        super().__init__()

    def set_database(self, new_database):
        super().set_database(new_database)

    def set_minimum_support(self, new_min_sup):
        super().set_minimum_support(new_min_sup)

    def run(self):
        if self._able_to_run is True:
            self._results = run_gpu_fast_itemset_miner(self._database, self._min_supp, self._gpu_setter)
        else:
            pass

    def get_results(self):
        return self._results


# if __name__ == '__main__':
#     a = Apriori()
#     db = get_database('retail.dat')
#     a.set_database(db)
#     a.set_minimum_support(10)
#     ts = timer()
#     a.run()
#     te = timer()
#     print(a.get_results(), te - ts)

from tkinter import *
from tkinter.ttk import *
from threading import Thread
from locals import *
from tool_bag import pick_algorithm, get_names_file
from pyDMLGPU.helper import get_database
from timeit import default_timer as timer
import pyopencl


class Worker(Thread):
    def __init__(self, gui_root, gui_result_box):
        Thread.__init__(self)
        self._gui_root = gui_root
        self._gui_result_box = gui_result_box
        # var which contains selected type
        self._algorithm_type = CHOSE_ALGORITHM
        # var which contains selected minimum support
        self._minimum_support = None
        # var which contains selected database file
        self._database = None
        # loop boolean
        self._loop = True
        # signal receiver
        self._signal_receiver = False
        # status worker -> state=True if working
        self._state = False
        # dictionary which holds names for each distinct code
        self._names_dictionary = None
        # flag event to be able to kill thread
        self._stop = True

    # write message info in gui's result box
    def __write_log(self, message, style):
        self._gui_result_box.config(state=NORMAL)
        self._gui_result_box.insert(END, message, style)
        self._gui_result_box.config(state=DISABLED)

    # process the database given
    def __start_processing(self):
        # start the progress bar
        working_bar = Progressbar(self._gui_root, orient=HORIZONTAL, length=300)
        working_bar.pack()
        working_bar.config(mode='indeterminate')
        working_bar.start()

        # processing block
        try:
            self.__write_log(START_LOG, STANDARD_FONT_COLOR)
            tool = pick_algorithm(self._algorithm_type)
            # Start processing with minimum support and algorithm type
            tool.set_database(self._database)
            tool.set_minimum_support(self._minimum_support)

            start_timing = timer()
            tool.run()
            end_timing = timer()
            results = tool.get_results()

            # write results
            if self._names_dictionary is None:
                self._print_codes(results, end_timing - start_timing)
            else:
                self._print_names(results, end_timing - start_timing)

        except (TypeError, ValueError, IOError, pyopencl.LogicError, pyopencl.Error,
                pyopencl.MemoryError, pyopencl.RuntimeError) as e:
            self.__write_log(BOX_POINTER + str(e)+NEW_LINE, ERROR_FONT_COLOR)
            self.__write_log(READY_LOG, STANDARD_FONT_COLOR)
        finally:
            # at the end stop the progress bar and remove it
            working_bar.stop()
            working_bar.destroy()
            # change state
            self._state = False

    def _print_names(self, results, elapsed_time):
        self.__write_log(MINIMUM_SUPPORT_LOG + str(self._minimum_support) + PROCENT_CHAR_LOG + NEW_LINE, RESULTS_FONT_COLOR)
        self.__write_log(NUMBER_RESULTS_LOG + str(len(results)) + NEW_LINE, RESULTS_FONT_COLOR)
        self.__write_log(RESULTS_LOG, RESULTS_FONT_COLOR)
        results_log = ''
        for item_set in results:
            if self._stop is True:
                log_item = log_item_set(self._names_dictionary, item_set, results.index(item_set))
                results_log += log_item

        if self._stop is True:
            self.__write_log(results_log, RESULTS_FONT_COLOR)
            self.__write_log(log_time_elapsed(elapsed_time), RESULTS_FONT_COLOR)
            self.__write_log(END_LOG, STANDARD_FONT_COLOR)
            self.__write_log(READY_LOG, STANDARD_FONT_COLOR)

    def _print_codes(self, results, elapsed_time):
        self.__write_log(MINIMUM_SUPPORT_LOG + str(self._minimum_support) + PROCENT_CHAR_LOG + NEW_LINE, RESULTS_FONT_COLOR)
        self.__write_log(NUMBER_RESULTS_LOG + str(len(results)) + NEW_LINE, RESULTS_FONT_COLOR)
        self.__write_log(RESULTS_LOG, RESULTS_FONT_COLOR)
        results_log = ''
        for item_set in results:
            if self._stop is True:
                results_log += TAB + TAB + str(item_set) + NEW_LINE

        if self._stop is True:
            self.__write_log(results_log, RESULTS_FONT_COLOR)
            self.__write_log(log_time_elapsed(elapsed_time), RESULTS_FONT_COLOR)
            self.__write_log(END_LOG, STANDARD_FONT_COLOR)
            self.__write_log(READY_LOG, STANDARD_FONT_COLOR)

    def run(self):
        while self._loop:
            self._stop = True
            if self._signal_receiver is True and self._state is False:
                self._signal_receiver = False
                self._state = True
                self.__start_processing()

    def set_algorithm_type(self, algorithm_type):
        self._algorithm_type = algorithm_type

    def set_minimum_support(self, minimum_support):
        self._minimum_support = minimum_support

    def set_database_file_path(self, path):
        self._database = get_database(path)

    def set_names(self, path):
        self._names_dictionary = get_names_file(path)

    def request_processing(self):
        if self._state is False:
            if self._database is not None and self._minimum_support is not None:
                self._signal_receiver = True
                return True
            return False
        else:
            self.__write_log(LOG_INFORMATIVE_WORKER_STATE, STANDARD_FONT_COLOR)
            return False

    def free_database(self):
        self._database = None

    def free_names(self):
        self._names_dictionary = None

    def kill(self):
        self._loop = False

    def stop(self):
        self._stop = False

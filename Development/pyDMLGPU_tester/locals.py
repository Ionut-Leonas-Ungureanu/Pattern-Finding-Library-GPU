
# result box locals
ID = 'Item-set_ID'
ITEM_SET_X_VALUE = 'x'
NEW_LINE = '\n'
BOX_POINTER = '>> '
TAB = '\t'
READY_LOG = BOX_POINTER + 'Ready...' + NEW_LINE
START_LOG = BOX_POINTER + 'Processing started...' + NEW_LINE
END_LOG = BOX_POINTER + 'Processing finished' + NEW_LINE
RESULTS_LOG = TAB + 'Results: ' + NEW_LINE
NUMBER_RESULTS_LOG = TAB + 'Number results: '
MINIMUM_SUPPORT_LOG = TAB + 'Minimum support: '
PROCENT_CHAR_LOG = '%'
TIME_UNIT_LOG = ' seconds'
ITEM_SET_TAG_LOG = 2 * TAB + 'Item-set ' + ID + ': '
ITEM_SET_VALUE_X_LOG = 3 * TAB + '---> ' + ITEM_SET_X_VALUE

# error log
ERROR_VALUE_SUPPORT = 'Error: minimum support must have values between 0 and 1, but not 0 and 1.' + NEW_LINE
ERROR_INPUT_FIELDS = 'Error: input fields are mandatory.' + NEW_LINE
ERROR_MIN_SUPP_VALUE = 'Error: minimum support field must be completed of float type.' + NEW_LINE

# messages
LOG_INFORMATIVE_WORKER_STATE = 'Worker::informative: I am working!' + NEW_LINE

# style names
STANDARD_FONT_COLOR = 'WHITE_FONT_COLOR'
RESULTS_FONT_COLOR = 'GREEN_FONT_COLOR'
ERROR_FONT_COLOR = 'RED_FONT_COLOR'

# algorithm types
CHOSE_ALGORITHM = 'Chose algorithm...'
PARALLEL_APRIORI_ALGORITHM = 'APRIORI'
PARALLEL_MINER_ALGORITHM = 'FAST ITEM MINER'

# colors
RESULT_COLOR = '#66ff66'
STANDARD_COLOR = 'white'
ERROR_COLOR = '#ff0000'
BACKGROUND_FRAME_COLOR = '#a3a3c2'
BACKGROUND_BUTTON_COLOR = '#3399ff'

# header
APP_TITLE = 'Data Mining Tool'


def log_time_elapsed(seconds):
    return TAB + 'Time: ' + str(seconds) + TIME_UNIT_LOG + NEW_LINE


def log_item_set(names, item_set, id_of):
    log = 2 * TAB + 'Item-set_' + str(id_of) + ': ' + NEW_LINE
    print('itemset', item_set)
    for item in item_set:
        if item in names.keys():
            name = names[item]
            log += 3 * TAB + '---> ' + name + NEW_LINE
        else:
            log += 3 * TAB + '---> ' + item + NEW_LINE
    return log

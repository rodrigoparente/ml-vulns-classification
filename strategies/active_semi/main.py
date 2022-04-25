# python imports
from timeit import default_timer as timer
from time import gmtime
from time import strftime

# third-party imports
import numpy as np

# project imports
from commons.file import to_file
from commons.file import fmt_list
from commons.data import load_data
from commons.telegram import send_message
from commons.classifiers import initial_pool_test_split
from commons.classifiers import unwrapping

# local imports
from .constants import LABELLED_CSV
from .constants import UNLABELLED_CSV
from .constants import OUTPUT_ACTIVE_SEMI
from .constants import LOG_ACTIVE_SEMI
from .utils import run_active_semi

# filtering messages to error
import warnings
warnings.filterwarnings('ignore')


def active_semi(classifiers, initial_size, test_size, n_repetitions, n_queries):

    unlabelled, _ = load_data(UNLABELLED_CSV)
    X, y = load_data(LABELLED_CSV)

    # setting random seed to make
    # sure the results are the same
    np.random.seed(42)

    initial_list, pool_list, test_list =\
        initial_pool_test_split(X, y, initial_size, test_size,
                                n_repetitions, init_split_method='kmedoids')

    for (model_name, scale_data) in classifiers:

        msg = f'# Begining test for {model_name.upper()} classifier.\n'
        to_file(LOG_ACTIVE_SEMI, msg)

        start = timer()

        for i, (initial_tuple, pool_tuple, test_tuple) in \
                enumerate(zip(initial_list, pool_list, test_list)):

            # copying original array
            X_unlabelled = unlabelled.copy()

            # safing unwrapping and copying values
            X_initial, X_pool, X_test, y_initial, y_pool, y_test =\
                unwrapping(initial_tuple, pool_tuple, test_tuple)

            metrics = run_active_semi(model_name, scale_data, X_unlabelled, X_initial,
                                      y_initial, X_pool, y_pool, X_test, y_test, n_queries)

            # saving metrics to file
            for key, values in metrics.items():
                values = fmt_list(values, "-") if key == 'cm' else fmt_list(values)
                to_file(f'{OUTPUT_ACTIVE_SEMI}/{model_name}-{key}.txt', f'{values}\n')

            if i % 25 == 0:
                msg = f'  - {i}% completed.\n'
                to_file(LOG_ACTIVE_SEMI, msg)

        msg = f'\n  Elapsed time: {strftime("%H:%M:%S", gmtime(timer() - start))}\n'
        to_file(LOG_ACTIVE_SEMI, msg)

        msg = f'Finished active semi-supervised learning: {model_name.upper()} test.'
        send_message(msg)

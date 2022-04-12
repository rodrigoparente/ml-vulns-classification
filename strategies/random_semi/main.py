# python imports
from timeit import default_timer as timer
from time import gmtime
from time import strftime

# third-party imports
import numpy as np

# local imports
from commons.file import to_file
from commons.file import metrics_to_file
from commons.data import load_data
from commons.telegram import send_message

# local imports
from .constants import LABELLED_CSV
from .constants import UNLABELLED_CSV
from .constants import OUTPUT_SEMI
from .constants import LOG_SEMI
from .utils import run_semi

# filtering messages to error
import warnings
warnings.filterwarnings('ignore')


def random_semi(classifiers, initial_size, test_size, n_repetitions, n_queries):

    X_unlabelled, _ = load_data(UNLABELLED_CSV)
    X, y = load_data(LABELLED_CSV)

    for (model_name, scale_data) in classifiers:

        # setting random seed to make
        # sure the results are the same
        np.random.seed(42)

        msg = f'# Begining test for {model_name.upper()} classifier.\n'
        to_file(LOG_SEMI, msg)

        start = timer()

        for i in range(n_repetitions):
            metrics = run_semi(model_name, scale_data, X_unlabelled,
                               X, y, initial_size, test_size, n_queries)

            output_url = f'{OUTPUT_SEMI}/{model_name}'
            metrics_to_file(output_url, metrics)

            if i % 25 == 0:
                msg = f'  - {i}% completed.\n'
                to_file(LOG_SEMI, msg)

        msg = f'\n  Elapsed time: {strftime("%H:%M:%S", gmtime(timer() - start))}\n'
        to_file(LOG_SEMI, msg)

        # send notification to telegram
        msg = f'Finished random semi-supervised learning: {model_name.upper()} test.'
        send_message(msg)

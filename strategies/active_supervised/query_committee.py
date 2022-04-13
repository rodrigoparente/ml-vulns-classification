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

# local imports
from .constants import LABELLED_CSV
from .constants import OUTPUT_STRATEGIES
from .constants import LOG_STRATEGIES
from .constants import QUERY_STRATEGIES
from .constants import COMMITTEE_STRATEGIES
from .constants import NO_COMMITTEE
from .utils import run_active_super

# filtering messages to error
import warnings
warnings.filterwarnings('ignore')


def query_committee(classifiers, initial_size, test_size, n_repetitions, n_queries, n_committee):

    X, y = load_data(LABELLED_CSV)

    for (model_name, scale_data) in classifiers:

        msg = f'# Begining test for {model_name.upper()} classifier.\n'
        to_file(LOG_STRATEGIES, msg)

        start = timer()

        names = ['query', 'committee']
        strategies = [QUERY_STRATEGIES, COMMITTEE_STRATEGIES]
        committees = [NO_COMMITTEE, n_committee]

        for strategy_name, strategy_list, committee_size in\
                zip(names, strategies, committees):

            # setting random seed to make
            # sure the results are the same
            np.random.seed(42)

            msg = f'  Testing {strategy_name} strategies...\n'
            to_file(LOG_STRATEGIES, msg)

            for i in range(n_repetitions):

                for strategy in strategy_list:
                    metrics = run_active_super(model_name, scale_data, strategy, committee_size,
                                               X, y, initial_size, test_size, n_queries)

                    # saving metrics to file
                    url = f'{OUTPUT_STRATEGIES}/{strategy_name}/{model_name}-{strategy}'

                    for key, values in metrics.items():
                        values = fmt_list(values, "-") if key == 'cm' else fmt_list(values)
                        to_file(f'{url}-{key}.txt', f'{values}\n')

                if i % 25 == 0:
                    msg = f'   - {i}% completed.\n'
                    to_file(LOG_STRATEGIES, msg)

        msg = f'\n  Elapsed time: {strftime("%H:%M:%S", gmtime(timer() - start))}\n'
        to_file(LOG_STRATEGIES, msg)

        # send notification to telegram
        msg = f'Finished active supervised query & committee: {model_name.upper()} test.'
        send_message(msg)

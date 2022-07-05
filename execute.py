# project imports
from strategies.active_supervised import query_committee
from strategies.active_semi import active_semi
from strategies.active_supervised import active_super
from strategies.random_semi import random_semi
from strategies.random_supervised import random_super

# local imports
from constants import CLASSIFIERS


def execute():

    INITIAL_SIZE = 20
    TEST_SIZE = 40
    N_REPETITIONS = 30
    N_QUERIES = 80
    N_COMMITTEE = 2

    print('Executing active learning query & committee strategies tests...')
    query_committee(classifiers=CLASSIFIERS,
                    initial_size=INITIAL_SIZE,
                    test_size=TEST_SIZE,
                    n_repetitions=N_REPETITIONS,
                    n_queries=N_QUERIES,
                    n_committee=N_COMMITTEE)

    print('Executing active semi-supervised learning tests...')
    active_semi(classifiers=CLASSIFIERS,
                initial_size=INITIAL_SIZE,
                test_size=TEST_SIZE,
                n_repetitions=N_REPETITIONS,
                n_queries=N_QUERIES)

    print('Executing active supervised learning tests...')
    active_super(classifiers=CLASSIFIERS,
                 initial_size=INITIAL_SIZE,
                 test_size=TEST_SIZE,
                 n_repetitions=N_REPETITIONS,
                 n_queries=N_QUERIES)

    print('Executing random supervised learning tests...')
    random_super(classifiers=CLASSIFIERS,
                 initial_size=INITIAL_SIZE,
                 test_size=TEST_SIZE,
                 n_repetitions=N_REPETITIONS,
                 n_queries=N_QUERIES)

    print('Executing random semi-supervised learning tests...')
    random_semi(classifiers=CLASSIFIERS,
                initial_size=INITIAL_SIZE,
                test_size=TEST_SIZE,
                n_repetitions=N_REPETITIONS,
                n_queries=N_QUERIES)

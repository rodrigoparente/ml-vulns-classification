# project imports
from strategies.active_supervised import query_committee
from strategies.active_semi import active_semi
from strategies.active_supervised import active_super
from strategies.random_semi import random_semi
from strategies.random_supervised import random_super

# local imports
from constants import CLASSIFIERS


def execute():

    print('Executing active learning query & committee strategies tests...')
    query_committee(classifiers=CLASSIFIERS,
                    initial_size=20,
                    test_size=40,
                    n_repetitions=30,
                    n_queries=80,
                    n_committee=2)

    print('Executing active semi-supervised learning tests...')
    active_semi(classifiers=CLASSIFIERS,
                initial_size=20,
                test_size=40,
                n_repetitions=30,
                n_queries=80)

    print('Executing active supervised learning tests...')
    active_super(classifiers=CLASSIFIERS,
                 initial_size=20,
                 test_size=40,
                 n_repetitions=30,
                 n_queries=80)

    print('Executing random supervised learning tests...')
    random_super(classifiers=CLASSIFIERS,
                 initial_size=20,
                 test_size=40,
                 n_repetitions=30,
                 n_queries=80)

    print('Executing random semi-supervised learning tests...')
    random_semi(classifiers=CLASSIFIERS,
                initial_size=20,
                test_size=40,
                n_repetitions=30,
                n_queries=80)

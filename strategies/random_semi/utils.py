# python imports
from timeit import default_timer as timer
from random import randrange

# third-party imports
import numpy as np

from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

# project imports
from commons.file import fmt_list
from commons.file import fmt_matrix
from commons.classifiers import get_estimator


def run_semi(model_name, scale_data, X_unlabelled,
             X, y, initial_size, test_size, n_queries):

    metrics = {
        'acc': list(),
        'time': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'cm': list()
    }

    X_pool, X_initial, y_pool, y_initial =\
        train_test_split(X, y, test_size=initial_size)

    X_pool, X_test, y_pool, y_test =\
        train_test_split(X_pool, y_pool, test_size=test_size)

    X_train = np.append(X_initial, X_unlabelled, axis=0)
    y_train = np.append(y_initial, np.full(X_unlabelled.shape[0], -1, dtype=int))

    X_train, y_train = shuffle(X_train, y_train)

    if scale_data:
        scaler = StandardScaler().fit(np.r_[X_train, X_pool])
        X_train = scaler.transform(X_train)
        X_pool = scaler.transform(X_pool)
        X_test = scaler.transform(X_test)

    for _ in range(n_queries):

        # select random item and append to X_train
        idx = randrange(len(X_pool))

        X_train = np.append(X_train, [X_pool[idx]], axis=0)
        y_train = np.append(y_train, [y_pool[idx]])

        X_train, y_train = shuffle(X_train, y_train)

        t_learn = timer()

        learner = SelfTrainingClassifier(
            base_estimator=get_estimator(model_name), threshold=0.99)

        learner.fit(X_train, y_train)

        t_learn = timer() - t_learn

        # delete selected item from X_pool

        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)

        # calculating metrics

        metrics['acc'].append(learner.score(X_test, y_test))
        metrics['time'].append(t_learn)

        y_pred = learner.predict(X_test)

        metrics['precision'].append(precision_score(y_test, y_pred, average='micro'))
        metrics['recall'].append(recall_score(y_test, y_pred, average='micro'))
        metrics['f1'].append(f1_score(y_test, y_pred, average='micro'))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3], normalize='true')
        metrics['cm'].append(fmt_list(fmt_matrix(cm)))

    return metrics

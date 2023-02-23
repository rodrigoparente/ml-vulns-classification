# python imports
from timeit import default_timer as timer

# third-party imports
import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# project imports
from commons.classifiers import get_estimator
from commons.file import fmt_matrix
from commons.file import fmt_list
from constants import LABELS


def run_active_semi(model_name, scale_data, X_unlabelled, X_initial,
                    y_initial, X_pool, y_pool, X_test, y_test, n_queries):

    metrics = {
        'acc': list(),
        'time': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'cm': list()
    }

    # creating X_train set
    X_train = X_initial
    y_train = y_initial

    if scale_data:
        scaler = StandardScaler().fit(np.r_[X_unlabelled, X_train, X_pool])
        X_unlabelled = scaler.transform(X_unlabelled)
        X_train = scaler.transform(X_train)
        X_pool = scaler.transform(X_pool)
        X_test = scaler.transform(X_test)

    learner_active = ActiveLearner(estimator=get_estimator(model_name),
                                   query_strategy=uncertainty_sampling,
                                   X_training=X_train, y_training=y_train)

    for _ in range(n_queries):
        # active learning

        query_idx, query_inst = learner_active.query(X_pool)

        X_train = np.append(X_train, query_inst, axis=0)
        y_train = np.append(y_train, y_pool[query_idx])

        learner_active.teach(query_inst.reshape(1, -1), y_pool[query_idx])

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        # semi-supervised learning

        X_labelled_unlabelled = np.append(X_train, X_unlabelled, axis=0)
        y_labelled_unlabelled = np.append(y_train, np.full(X_unlabelled.shape[0], -1, dtype=int))

        X_labelled_unlabelled, y_labelled_unlabelled =\
            shuffle(X_labelled_unlabelled, y_labelled_unlabelled)

        t_learn = timer()

        learner_semi = SelfTrainingClassifier(
            base_estimator=get_estimator(model_name), threshold=0.99)

        learner_semi.fit(X_labelled_unlabelled, y_labelled_unlabelled)

        t_learn = timer() - t_learn

        # calculating metrics

        metrics['acc'].append(learner_semi.score(X_test, y_test))
        metrics['time'].append(t_learn)

        y_pred = learner_semi.predict(X_test)

        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))

        cm = confusion_matrix(y_test, y_pred, labels=LABELS, normalize='true')
        metrics['cm'].append(fmt_list(fmt_matrix(cm)))

    return metrics

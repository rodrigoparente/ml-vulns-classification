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

# project imports
from strategies.active_supervised.utils import initial_train_test_split


def run_active_semi(model_name, scale_data, X_unlabelled,
                    X, y, initial_size, test_size, n_queries):

    accs, times, precisions, recalls, f1s, cms = (list() for _ in range(6))

    X_initial, X_pool, X_test, y_initial, y_pool, y_test =\
        initial_train_test_split(X, y, initial_size, test_size)

    if scale_data:
        scaler = StandardScaler().fit(np.r_[X_pool, X_initial])
        X_initial = scaler.transform(X_initial)
        X_pool = scaler.transform(X_pool)
        X_test = scaler.transform(X_test)

    learner_active = ActiveLearner(estimator=get_estimator(model_name),
                                   query_strategy=uncertainty_sampling,
                                   X_training=X_initial, y_training=y_initial)

    X_train, y_train = X_initial, y_initial

    for _ in range(n_queries):
        # active learning

        query_idx, query_inst = learner_active.query(X_pool)

        X_train = np.append(X_train, query_inst, axis=0)
        y_train = np.append(y_train, y_pool[query_idx])

        learner_active.teach(query_inst.reshape(1, -1), y_pool[query_idx])

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

        # delete selected item from X_pool

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        # calculating metrics

        accs.append(learner_semi.score(X_test, y_test))
        times.append(t_learn)

        y_pred = learner_semi.predict(X_test)

        precisions.append(precision_score(y_test, y_pred, average='micro'))
        recalls.append(recall_score(y_test, y_pred, average='micro'))
        f1s.append(f1_score(y_test, y_pred, average='micro'))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3], normalize='true')
        cms.append(fmt_list(fmt_matrix(cm)))

    return [accs, times, precisions, recalls, f1s, cms]

# python imports
from timeit import default_timer as timer

# third-party imports
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from modAL.models import ActiveLearner
from modAL.models import Committee
from modAL.uncertainty import uncertainty_sampling
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import margin_sampling
from modAL.disagreement import consensus_entropy_sampling
from modAL.disagreement import max_disagreement_sampling
from modAL.disagreement import vote_entropy_sampling

from sklearn_extra.cluster import KMedoids

# project imports
from commons.classifiers import get_estimator
from commons.file import fmt_matrix
from commons.file import fmt_list


def initial_train_test_split(X, y, initial_size, test_size):

    X_pool, X_test, y_pool, y_test =\
        train_test_split(X, y, test_size=test_size, stratify=y)

    # calculating clusters centers
    kmedoids = KMedoids(n_clusters=initial_size)
    kmedoids.fit(StandardScaler().fit_transform(X_pool))

    # get the indexes of the medoids centers
    initial_idx = kmedoids.medoid_indices_

    # selecting elements to X_initial
    X_initial, y_initial = X_pool[initial_idx], y_pool[initial_idx]

    # removing selected elements from X_pool
    X_train = np.delete(X_pool, initial_idx, axis=0)
    y_train = np.delete(y_pool, initial_idx, axis=0)

    # shuffling data
    X_initial, y_initial = shuffle(X_initial, y_initial)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    return X_initial, X_train, X_test, y_initial, y_train, y_test


def get_query_strategy(name):
    if name == 'entropy-sampling':
        return entropy_sampling
    elif name == 'margin-sampling':
        return margin_sampling
    elif name == 'uncertainty-sampling':
        return uncertainty_sampling


def get_committee_strategy(name):
    if name == 'vote-entropy-sampling':
        return vote_entropy_sampling
    elif name == 'consensus-entropy-sampling':
        return consensus_entropy_sampling
    elif name == 'max-disagreement-sampling':
        return max_disagreement_sampling


def bagging(model_name, strategy_name, committee_size,
            X_initial, y_initial, initial_size):

    learners = list()

    for _ in range(committee_size):
        train_idx = np.random.choice(
            range(X_initial.shape[0]), size=initial_size, replace=True)

        X_train = X_initial[train_idx]
        y_train = y_initial[train_idx]

        # initializing learner
        learner = ActiveLearner(
            estimator=get_estimator(model_name),
            X_training=X_train, y_training=y_train,
            query_strategy=get_committee_strategy(strategy_name)
        )

        learners.append(learner)

    # assembling the committee
    return Committee(learner_list=learners)


def run_active_super(model_name, scale_data, strategy_name, committee_size,
                     X, y, initial_size, test_size, n_queries):

    accs, times, precisions, recalls, f1s, cms = (list() for _ in range(6))

    learner = None

    X_initial, X_pool, X_test, y_initial, y_pool, y_test =\
        initial_train_test_split(X, y, initial_size, test_size)

    if scale_data:
        scaler = StandardScaler().fit(np.r_[X_pool, X_initial])
        X_initial = scaler.transform(X_initial)
        X_pool = scaler.transform(X_pool)
        X_test = scaler.transform(X_test)

    if committee_size > 0:
        learner = bagging(model_name, strategy_name, committee_size,
                          X_initial, y_initial, initial_size)
    else:
        learner = ActiveLearner(estimator=get_estimator(model_name),
                                query_strategy=get_query_strategy(strategy_name),
                                X_training=X_initial, y_training=y_initial)

    for _ in range(n_queries):

        query_idx, query_inst = learner.query(X_pool)

        t_learn = timer()

        learner.teach(query_inst.reshape(1, -1), y_pool[query_idx])

        t_learn = timer() - t_learn

        # delete selected item from X_pool

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        # calculating metrics

        accs.append(learner.score(X_test, y_test))
        times.append(t_learn)

        y_pred = learner.predict(X_test)

        precisions.append(precision_score(y_test, y_pred, average='micro'))
        recalls.append(recall_score(y_test, y_pred, average='micro'))
        f1s.append(f1_score(y_test, y_pred, average='micro'))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3], normalize='true')
        cms.append(fmt_list(fmt_matrix(cm)))

    return [accs, times, precisions, recalls, f1s, cms]

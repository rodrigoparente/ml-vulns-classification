# python imports
from timeit import default_timer as timer

# third-party imports
import numpy as np

from sklearn.preprocessing import StandardScaler
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
from modAL.disagreement import KL_max_disagreement

# project imports
from commons.classifiers import get_estimator
from commons.file import fmt_matrix
from commons.file import fmt_list
from constants import LABELS


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
    elif name == 'kl-max-disagreement':
        return KL_max_disagreement


def bagging(model_name, strategy_name,
            committee_size, X_initial, y_initial):

    learners = list()

    for _ in range(committee_size):
        train_idx = np.random.choice(
            range(X_initial.shape[0]), size=X_initial.shape[0], replace=True)

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


def run_active_super(model_name, scale_data, strategy_name,
                     committee_size, X_initial, y_initial,
                     X_pool, y_pool, X_test, y_test, n_queries):

    metrics = {
        'acc': list(),
        'time': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'cm': list()
    }

    learner = None

    if scale_data:
        scaler = StandardScaler().fit(np.r_[X_initial, X_pool])
        X_initial = scaler.transform(X_initial)
        X_pool = scaler.transform(X_pool)
        X_test = scaler.transform(X_test)

    if committee_size > 0:
        learner = bagging(model_name, strategy_name,
                          committee_size, X_initial, y_initial)
    else:
        learner = ActiveLearner(estimator=get_estimator(model_name),
                                query_strategy=get_query_strategy(strategy_name),
                                X_training=X_initial, y_training=y_initial)

    for _ in range(n_queries):

        query_idx, query_inst = learner.query(X_pool)

        t_learn = timer()

        learner.teach(query_inst.reshape(1, -1), y_pool[query_idx])

        t_learn = timer() - t_learn

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        # calculating metrics

        metrics['acc'].append(learner.score(X_test, y_test))
        metrics['time'].append(t_learn)

        y_pred = learner.predict(X_test)

        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))

        cm = confusion_matrix(y_test, y_pred, labels=LABELS, normalize='true')
        metrics['cm'].append(fmt_list(fmt_matrix(cm)))

    return metrics

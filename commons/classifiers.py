# third-party imports
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn_extra.cluster import KMedoids


def get_estimator(name):
    if name == 'rf':
        return RandomForestClassifier()
    elif name == 'gb':
        return GridSearchCV(GradientBoostingClassifier(), param_grid={
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 5, 7],
            "n_estimators": [50, 100, 200]
        }, scoring='accuracy', cv=5, n_jobs=-1)
    elif name == 'lr':
        return LogisticRegression(penalty='none')
    elif name == 'svc':
        return GridSearchCV(SVC(probability=True), [
            {'kernel': ['rbf'], 'gamma': 2 ** np.arange(-15.0, 4.0, 2),
                'C': 2 ** np.arange(-5.0, 16.0, 2)},
            {'kernel': ['linear'], 'C': 2 ** np.arange(-5.0, 16.0, 2)},
            {'kernel': ['poly'], 'C': 2 ** np.arange(-5.0, 16.0, 2),
                'degree': [2, 3, 4, 5]}
        ], scoring='accuracy', cv=5, n_jobs=-1)
    elif name == 'mlp':
        return MLPClassifier()


def initial_pool_test_split(X, y, initial_size, test_size,
                            n_repetitions, init_split_method='none'):

    initial_list = list()
    pool_list = list()
    test_list = list()

    # splitting data into pool and test
    for _ in range(n_repetitions):

        X_pool, X_test, y_pool, y_test =\
            train_test_split(X, y, test_size=test_size)

        pool_list.append((X_pool, y_pool))
        test_list.append((X_test, y_test))

    # spliting pool into initial and pool
    for pool in pool_list:

        X_pool, y_pool = pool

        if init_split_method == 'kmedoids':
            # calculating clusters centers
            kmedoids = KMedoids(n_clusters=initial_size)
            kmedoids.fit(StandardScaler().fit_transform(X_pool))

            # get the indexes of the medoids centers
            initial_idx = kmedoids.medoid_indices_

            # selecting elements to X_initial
            X_initial, y_initial = X_pool[initial_idx], y_pool[initial_idx]

            # removing selected elements from X_pool
            X_pool = np.delete(X_pool, initial_idx, axis=0)
            y_pool = np.delete(y_pool, initial_idx, axis=0)
        else:
            X_pool, X_initial, y_pool, y_initial =\
                train_test_split(X_pool, y_pool, test_size=initial_size)

        initial_list.append((X_initial, y_initial))

    # shuffling data
    for (X_initial, y_initial), (X_pool, y_pool), (X_test, y_test) in \
            zip(initial_list, pool_list, test_list):

        X_initial, y_initial = shuffle(X_initial, y_initial)
        X_pool, y_pool = shuffle(X_pool, y_pool)
        X_test, y_test = shuffle(X_test, y_test)

    return initial_list, pool_list, test_list


def unwrapping(initial_tuple, pool_tuple, test_tuple):
    X_initial, y_initial = initial_tuple
    X_initial, y_initial = X_initial.copy(), y_initial.copy()

    X_pool, y_pool = pool_tuple
    X_pool, y_pool = X_pool.copy(), y_pool.copy()

    X_test, y_test = test_tuple
    X_test, y_test = X_test.copy(), y_test.copy()

    return X_initial, X_pool, X_test, y_initial, y_pool, y_test

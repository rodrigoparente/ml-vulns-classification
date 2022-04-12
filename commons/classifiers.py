# third-party imports
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_estimator(name):
    if name == 'rf':
        return RandomForestClassifier()
    elif name == 'gb':
        return GradientBoostingClassifier()
    elif name == 'lr':
        return LogisticRegression(penalty='none')
    elif name == 'svc':
        return GridSearchCV(SVC(probability=True, max_iter=2000000), [
            {'kernel': ['rbf'], 'gamma': 2 ** np.arange(-15.0, 4.0, 2),
                'C': 2 ** np.arange(-5.0, 16.0, 2)},
            {'kernel': ['linear'], 'C': 2 ** np.arange(-5.0, 16.0, 2)}
        ], scoring='accuracy', cv=5)
    elif name == 'svc-linear':
        return SVC(kernel='linear', C=2, probability=True, max_iter=2000000)
    elif name == 'mlp':
        return MLPClassifier()

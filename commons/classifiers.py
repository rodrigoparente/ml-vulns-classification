# third-party imports
import numpy as np

# patching sklearn lib to run faster
# https://intel.github.io/scikit-learn-intelex/what-is-patching.html
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.ensemble import GradientBoostingClassifier  # noqa E402
from sklearn.linear_model import LogisticRegression  # noqa E402
from sklearn.ensemble import RandomForestClassifier  # noqa E402
from sklearn.model_selection import GridSearchCV  # noqa E402
from sklearn.neural_network import MLPClassifier  # noqa E402
from sklearn.svm import SVC  # noqa E402


def get_estimator(name):
    if name == 'rf':
        return RandomForestClassifier()
    elif name == 'gb':
        return GradientBoostingClassifier()
    elif name == 'lr':
        return LogisticRegression(penalty='none')
    elif name == 'svc-with-grid':
        return GridSearchCV(SVC(probability=True, random_state=42), [
            {'kernel': ['rbf'], 'gamma': 2 ** np.arange(-15.0, 4.0, 2),
                'C': 2 ** np.arange(-5.0, 16.0, 2)},
            {'kernel': ['linear'], 'C': 2 ** np.arange(-5.0, 16.0, 2)}
        ], scoring='accuracy', cv=5, n_jobs=-1)
    elif name == 'svc':
        return SVC(kernel='linear', C=2, probability=True)
    elif name == 'mlp':
        return MLPClassifier()

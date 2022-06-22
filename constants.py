LABELS = [0, 1, 2, 3]
LABELS_DICT = {'LOW': 0, 'MODERATE': 1, 'IMPORTANT': 2, 'CRITICAL': 3}

CLASSIFIERS = [
    ('rf', False),
    ('gb', False),
    ('lr', True),
    ('svc', True),
    ('mlp', True)
]

INPUT_RAW_FILE_PATH = 'results/raw'
OUTPUT_FILE_PATH = 'results/compiled'

TECHNIQUES = ['active-semi', 'active-super', 'random-semi', 'random-super']

LEARNERS = ['rf', 'gb', 'lr', 'svc', 'mlp']

QUERY_METHOD = ['query', 'committee']

STRATEGIES = [
    ['entropy-sampling', 'margin-sampling', 'uncertainty-sampling'],
    ['consensus-entropy-sampling', 'max-disagreement-sampling', 'vote-entropy-sampling']
]

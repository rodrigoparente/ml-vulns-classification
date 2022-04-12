CLASSIFIERS = [
    ('rf', False),
    ('gb', False),
    ('lr', True),
    ('svc', False),
    ('mlp', True)
]

TECHNIQUES = ['active-semi', 'active-super', 'random-semi', 'random-super']

LEARNERS = ['rf', 'gb', 'lr', 'svc', 'mlp']

INPUT_RAW_FILE_PATH = 'results/raw'

OUTPUT_FILE_PATH = 'results/compiled'

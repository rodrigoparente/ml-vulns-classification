LABELLED_CSV = 'datasets/vulns-labelled.csv'

OUTPUT_STRATEGIES = 'results/raw/active-super-strategies'
OUTPUT_ACTIVE_SUPER = 'results/raw/active-super'

LOG_STRATEGIES = 'logs/active-super-strategies.txt'
LOG_ACTIVE_SUPER = 'logs/active-super.txt'

QUERY_STRATEGIES = [
    'entropy-sampling',
    'uncertainty-sampling'
]

COMMITTEE_STRATEGIES = [
    'kl-max-disagreement',
    'vote-entropy-sampling'
]

NO_COMMITTEE = 0

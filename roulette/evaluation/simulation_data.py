from collections import namedtuple

ExperimentData = namedtuple('ExperimentData', [
    "Real",
    'Model',
    'Rand',
    'Mean',
])

Score = namedtuple('Score', [
    'Model',
    'Rand',
    'Mean',
])

Metrics = namedtuple('Metrics',
                     [
                         'Discriminability',
                         'Certainty',
                     ]
                     )

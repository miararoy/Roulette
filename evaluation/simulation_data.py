from collections import namedtuple

ExperimentData = namedtuple('ExperimentData', [
    "Real",
    'Model',
    'Rand',
    'Mean',
    'OtherModels',
])

Score = namedtuple('Score', [
    'Model',
    'Rand',
    'Mean',
    'Div',
    'OtherModels',
])

Metrics = namedtuple('Metrics',
                     ['Discriminability', 'Certainty', 'Divergency'])

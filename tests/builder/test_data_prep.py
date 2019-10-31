import unittest
import pandas as pd
import numpy as np

from roulette.builder.data_prep import prepare_data_for_training


class DataPrepTestCase(unittest.TestCase):
    df = pd.DataFrame(
        [
            {"id": 1, "size": 3, "target": 0},
            {"id": 2, "size": 5, "target": 0},
            {"id": 3, "size": 10, "target": 1}
        ]
    )
    expected_index_df = pd.DataFrame(
        [
            {"size": 3},
            {"size": 5},
            {"size": 10}
        ],
        index=[1, 2, 3]
    )
    expected_df = pd.DataFrame(
        [
            {"id": 1, "size": 3},
            {"id": 2, "size": 5},
            {"id": 3, "size": 10}
        ]
    )
    expected_target = np.asarray([0, 0, 1])

    def test_not_indexed(self):
        a, t, _, _ = prepare_data_for_training(
            self.df, target="target", validation_test_size=0)
        pd.testing.assert_frame_equal(
            self.expected_df,
            a
        )
        np.testing.assert_array_almost_equal(t, self.expected_target)
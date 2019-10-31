import unittest

import numpy as np

from roulette.evaluation.norms import min_max_norm, get_normalizer


def test_min_max_norm():
    np.testing.assert_array_almost_equal(
        np.asarray([0, 0.5, 1.0]),
        min_max_norm([9, 10, 11])
    )
    np.testing.assert_array_almost_equal(
        np.asarray([0, 0.5, 1.0]),
        min_max_norm([-1.0, 0.0, 1.0])
    )


class NormalizerTestCase(unittest.TestCase):
    def test_get_normalizer(self):
        with self.assertRaises(KeyError) as context:
            get_normalizer("wrong_key")

        self.assertTrue(
            "normalizer is not in available" in str(
                context.exception))

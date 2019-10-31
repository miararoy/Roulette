from roulette.evaluation.utils import close_enough,\
    validate_multiple_lists_length,\
    samples_to_bin_numbers


def test_close_enough():
    a = 3.0002
    b = 3.00019
    assert close_enough(a, b)
    assert not close_enough(a, b, 5)
    assert close_enough(a / 100, b / 100, 5)


def test_validate_multiple_lists_length():
    assert validate_multiple_lists_length([1, 3, 4], [1, 4, 4], [2, 2, 2],
                                          [9, 9, 9])
    assert not validate_multiple_lists_length([1, 3, 4], [2, 2, 2], 4)
    assert not validate_multiple_lists_length([1, 3, 2], [2, 2], [2, 3, 4])


def test_sample_to_bin():
    bins = [0., 0.25, 0.5, 0.75, 1]
    a = [x / 10 for x in range(10)]
    b = [0.1] * 10
    n_a, n_b = samples_to_bin_numbers(a, b, bins=bins)
    assert n_b == [0] * 10
    assert n_a == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    c = [0.8] * 10
    _, _, n_c = samples_to_bin_numbers(a, b, c, bins=bins)
    assert n_c == [3] * 10

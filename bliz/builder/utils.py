import numpy as np

from collections import namedtuple
from sklearn.metrics import classification_report, confusion_matrix


Doc = namedtuple(
    'Doc',
    [
        "version",
        "type",
        "algo",
        "param",
        "cv",
    ]
)


def compress_regression_results(l, true_condition=lambda x: x >= 0.6):
    out = []
    for x in list(l):
        if true_condition(x):
            out.append(1)
        else:
            out.append(0)
    return out


def generate_model_documentation(
    model_version: str,
    model_type: str,
    model_algorithm: str,
    model_parameter_tuning: str = None,
    model_cv: str = None,
):
    return (
        "\n---------- Model Details:\n\n" +
        "Model Version == {}\n".format(model_version) +
        "Model Type == {}\n".format(model_type) +
        "Model Algorithm == {}\n".format(model_algorithm) +
        "Model Parameter Tuning == {}\n".format(model_parameter_tuning) +
        "Model CV == {}\n".format(model_cv)
    )


def create_classification_report(y_real, y_pred):
    p = compress_regression_results(list(y_pred))
    r = compress_regression_results(list(y_real))

    for y_p, y_r_p, y_r, y_r_r in zip(
        p,
        list(y_pred),
        r,
        list(y_real),
    ):
        print("Predicted {rp} ~ {p} for result {rr} ~ {r}".format(
            p=y_p,
            rp=y_r_p,
            r=y_r,
            rr=y_r_r,
        ))
    print("\n{}".format(
        classification_report(
            r,
            p,
            target_names=["bottom_tier", "top_tier"],
        )
    )
    )
    tn, fp, fn, tp = confusion_matrix(p, r).ravel()
    print(
        "tn = {} \n".format(tn / len(p)) +
        "tp = {} \n".format(tp / len(p)) +
        "fn = {} \n".format(fn / len(p)) +
        "fp = {} \n".format(fp / len(p))
    )
    print(
        "Precision = {}\n".format(round(tp / (tp + fp), 2)) +
        "Recall = {}\n".format(round(tp / (tp + fn), 2))
    )


def min_max_norm(y: np.ndarray) -> np.ndarray:
    return (y - y.min()) / (y.max() - y.min())


def is_regression_metric(metric: callable) -> bool:
    real = np.asarray([0.1, 0.33, 0.44])
    pred_close = np.asarray([0.11, 0.34, 0.45])
    pred_far = np.asarray([0.3, 0.6, 0.9])
    if (not metric(real, real) == 0.0 and
            not metric(real, real) < metric(real, pred_close) < metric(real, pred_far)):
        return False
    else:
        return True


def is_binary_classification_metric(metric: callable) -> bool:
    real = np.asarray([1, 1, 0])
    pred_close = np.asarray([1, 0, 0])
    pred_far = np.asarray([0, 0, 1])
    if (not metric(real, real) == 0.0 and
            not metric(real, real) < metric(real, pred_close) < metric(real, pred_far)):
        return False
    else:
        return True
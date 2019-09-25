import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bliz.builder.utils import min_max_norm
from bliz.logger import Logger


def create_evaluation_plot_with_rp(y: np.ndarray, y_pred: np.ndarray):
    _y_pred = min_max_norm(y_pred)
    logger = Logger("evaluation plot").get_logger()
    val_pred = pd.concat([pd.Series(y),pd.Series(_y_pred)],axis=1)
    val_pred.columns = ['test','pred']


    total_points = len(y)
    tp = val_pred[(val_pred.test>0.5)&(val_pred.pred>0.5)].shape[0]
    tn = val_pred[(val_pred.test<0.5)&(val_pred.pred<0.5)].shape[0]
    fn = val_pred[(val_pred.test>0.5)&(val_pred.pred<0.5)].shape[0]
    fp = val_pred[(val_pred.test<0.5)&(val_pred.pred>0.5)].shape[0]
    try:
        recall = round(tp/(tp+fn),2)
    except ZeroDivisionError:
        logger.warn("devision by zero in calcaulatin recall")
        recall = -1.0
    try:
        precision = round(tp/(tp+fp),2)
    except ZeroDivisionError:
        logger.warn("devision by zero in calcaulatin precision")
        precision = -1.0
    try:
        n_precision = round(tn/(tn+fn),2)
    except ZeroDivisionError:
        logger.warn("devision by zero in calcaulatin precision")
        precision = -1.0
    try:
        specificity = round(tn/(tn+fp),2)
    except ZeroDivisionError:
        logger.warn("devision by zero in calcaulatin specificity")
        precision = -1.0

    sns_plot = sns.regplot(
        y=_y_pred, 
        x=y,
        color="skyblue",
        fit_reg=False
    ).set_title(label="test set: recall = {rec}, precision = {pre}, specificity = {specificity}, n_precision = {n_pre}, N = {n}".format(
        rec=recall, pre=precision, specificity=specificity, n_pre=n_precision, n=total_points
    ),
        fontdict={'fontsize': 10}
    )

    axes = sns_plot.axes
    axes.set_ylim(0,1)
    axes.set_xlim(0,1)
    axes.set(xlabel='real', ylabel='prediction')

    axes.axhline(0.5, ls='--', c="r")
    axes.axvline(0.5, ls='--', c='r')
    plt.text(0.75, 0.75, round(tp/total_points,2), color='r')
    plt.text(0.2, 0.75, round(fp/total_points,2), color='r')
    plt.text(0.75, 0.2, round(fn/total_points,2), color='r')
    plt.text(0.2, 0.2, round(tn/total_points,2), color='r')
    return plt, precision, recall, n_precision, specificity
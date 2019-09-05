import os

import matplotlib.pyplot as plt
import seaborn as sns


def save_multiple_hist(
        data: list,
        path: str,
        bins: list,
):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        for d in data:
            if d._fields == ('name', 'data', 'color'):
                plot_name = "plot_{}_multi.png".format(d.name)
                plot = sns.distplot(
                    d.data,
                    bins=bins,
                    color=d.color,
                )
                plot.set(ylim=(0, len(d.data) / 10))
                plot_path = os.path.join(path, plot_name)
                print("saving... {}".format(plot_path))
                plot.figure.savefig(plot_path, transparent=True, dpi=300)
                plot.clear()
            else:
                raise ValueError("data should be of type RestultData")
    except Exception as e:
        print("save_multiple_hist FAILED")
        raise e


def save_single_hist(
        data: list,
        path: str,
        bins: list,
        name: str = "model",
):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        plot_name = "{}_plot.png".format(name)
        for d in data:
            if d._fields == ('name', 'data', 'color'):
                plot = sns.distplot(
                    d.data,
                    bins=bins,
                    color=d.color,
                    kde_kws={"label": d.name})
                plot_path = os.path.join(path, plot_name)
            else:
                raise ValueError("data should be of type RestultData")
        print("saving... {}".format(plot_path))
        plot.set_xlim(0,0.5)
        plot.figure.savefig(plot_path, dpi=300)
    except Exception as e:
        print("save_single_hist FAILED")
        raise e

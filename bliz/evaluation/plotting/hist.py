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


def single_hist(
        data: list,
        bins: list,
        path: str = None,
        title: str = None,
        name: str = "model",
):
    try:
        for d in data:
            if d._fields == ('name', 'data', 'color'):
                plot = sns.distplot(
                    d.data,
                    bins=bins,
                    color=d.color,
                    kde_kws={"label": d.name})
                if title:
                    plot.set_title(title)
            else:
                raise ValueError("data should be of type RestultData")
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            plot_name = "{}_plot.png".format(name)
            plot_path = os.path.join(path, plot_name)
            print("saving... {}".format(plot_path))
            plot.figure.savefig(plot_path, dpi=300)
        else:
            return plot
    except Exception as e:
        print("save_single_hist FAILED")
        raise e


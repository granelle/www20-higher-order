from __future__ import print_function

import os
from collections import defaultdict

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import mpltex
import numpy as np
import pandas as pd
import pylab
from matplotlib.offsetbox import AnchoredText
from src.config import *
from src.config import *

params = {'legend.fontsize': 'x-large',
          'axes.titlesize': 14,
          'axes.labelsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10}
pylab.rcParams.update(params)


def get_agg_df(feature, hs, neg_type, imb, seed=0):
    from src.config import results_dir

    # agg dir

    results_dir = results_dir.format(hs, neg_type, imb, seed)

    agg_dir = results_dir + 'agg/'
    agg_path = agg_dir + '{}-aggregate.csv'.format(feature)
    agg_df = pd.read_csv(agg_path, index_col=0)

    return agg_df


def _get_feature_mean_df(hs, neg_type, imb, seed):
    dfs = []

    for feature in features:
        agg_df = get_agg_df(feature, hs, neg_type, imb, seed)
        dfs.append(agg_df)

    concat_df = pd.concat(dfs)

    mean_df = concat_df.groupby(level=0).mean()

    return mean_df


def _plot_mi_correlation(axes, neg_type, imb, seed, col_idx, hs, which_mi):
    mean_df = _get_feature_mean_df(hs, neg_type, imb, seed)

    mi_df = pd.read_csv(statistics_dir + "mis_exp.csv", index_col=0)

    dots_x = []
    dots_y = []

    for gn in gns_display:

        dotstyle = dotstyle_map[gn]

        pr_row = mean_df.loc[gn]
        mi_row = mi_df.loc[gn]

        if which_mi == "32":
            ax = axes[col_idx]

            pr = pr_row.loc["%PR2"]
            mi = mi_row.loc["mi_32"]

            ax.plot(mi,
                    pr,
                    label='{}'.format(gn),
                    **dotstyle)

            ax.tick_params(axis='both', which='major', labelsize=7)

            ax.set_xlabel("I(3;2)")
            ax.set_ylabel("2 to 3 gain (%)", rotation='vertical', labelpad=1)

            dots_x.append(mi)
            dots_y.append(pr)

            ax.set_title("Size {} prediction".format(hs), fontsize=10, fontweight="bold")

        elif which_mi == "42":
            assert (hs == 4)
            ax = axes

            pr = pr_row.loc["PR1"]
            mi = mi_row.loc["mi_42"]

            ax.plot(mi,
                    pr,
                    label='{}'.format(gn),
                    **dotstyle)

            ax.tick_params(axis='both', which='major', labelsize=7)

            ax.set_xlabel("I(4;2)")
            ax.set_ylabel("2-order AUC-PR", rotation='vertical', labelpad=1)

            dots_x.append(mi)
            dots_y.append(pr)

            ax.set_title("Size {} prediction".format(hs), fontsize=10, fontweight="bold")

    cc = np.corrcoef(dots_x, dots_y)[0][1]

    anc = AnchoredText("CorrCoef: {}".format(round(cc, 2)), loc="upper right", frameon=True)
    ax.add_artist(anc)
    ax.add_artist(anc)

    if which_mi == "32":

        if neg_type == "clique":

            if imb == 10:
                scale = 10
                shift = 0

            if imb == 5:
                scale = 10
                shift = 0.2

            if imb == 2:
                scale = 5
                shift = -4

            elif imb == 1:
                scale = 2.6
                shift = -4

            x = np.linspace(scale / float(max(dots_y) - shift), 1.99, 100)
            y = scale * np.reciprocal(x) + shift
            ax.plot(x, y, color='black', linestyle=':')

    elif which_mi == "42":

        if neg_type == "clique":
            if imb == 10:
                scale = 1 / 15.0
                shift = 0.2

            if imb == 5:
                scale = 1 / 15.0
                shift = 0.3

            if imb == 2:
                scale = 1 / 20.0
                shift = 0.5

            elif imb == 1:
                scale = 1 / 30.0
                shift = 0.65

            x = np.linspace(scale / float(max(dots_y) - shift), 1.9, 100)
            y = scale * np.reciprocal(x) + shift
            ax.plot(x, y, color='black', linestyle=':')

    return axes


def plot_mi32_correlations(neg_type, imb, seed=0):
    fig, axes = plt.subplots(1, 2, sharex=False, sharey=False)
    fig.set_size_inches((3 * 2, 3 * 1))

    axes = _plot_mi_correlation(axes, neg_type, imb, seed, col_idx=0, hs=4, which_mi="32")
    axes = _plot_mi_correlation(axes, neg_type, imb, seed, col_idx=1, hs=5, which_mi="32")

    plt.tight_layout()

    plots_path = plots_dir + "mi/neg-{}{}_seed-{}_MI32-CORRELATIONS".format(neg_type, imb, seed)
    fig.savefig(plots_path + '.pdf', bbox_inches='tight', pad_inches=0.03)

    print(plots_path)

    fig.show()


def plot_mi42_correlations(neg_type, imb, seed=0):
    fig, axes = plt.subplots(1, 1, sharex=False, sharey=False)
    fig.set_size_inches((3 * 1, 3 * 1))

    axes = _plot_mi_correlation(axes, neg_type, imb, seed, col_idx=0, hs=4, which_mi="42")

    plt.tight_layout()

    plots_path = plots_dir + "mi/neg-{}{}_seed-{}_MI42-CORRELATIONS".format(neg_type, imb, seed)
    fig.savefig(plots_path + '.pdf', bbox_inches='tight', pad_inches=0.03)

    print(plots_path)

    fig.show()


def _plot_subdots(axes, hs, neg_type, imb, seed):
    mean_df = _get_feature_mean_df(hs, neg_type, imb, seed)

    print(mean_df)

    for gn in gns_display:

        dotstyle = dotstyle_map[gn]

        pr_row = mean_df.loc[gn]

        if hs == 4:
            for col_idx, p in [(0, 2)]:
                col_idx = 0

                ax = axes[col_idx]
                ax.plot(pr_row.loc["PR{}".format(p - 1)],
                        pr_row.loc["PR{}".format(p)],
                        label='{}'.format(gn),
                        **dotstyle)
                ax.plot([0, 1], [0, 1], linestyle=':', color='dodgerblue')

                ax.set_xlabel("{}-order AUC-PR".format(p))
                ax.set_ylabel("{}-order AUC-PR".format(p + 1), labelpad=3)

                ax.xaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.tick_params(axis='both', which='major')
                ax.set(adjustable='box-forced', aspect='equal')

        elif hs == 5:

            for col_idx, p in [(1, 2), (2, 3)]:
                ax = axes[col_idx]

                ax.plot(pr_row.loc["PR{}".format(p - 1)],
                        pr_row.loc["PR{}".format(p)],
                        label='{}'.format(gn),
                        **dotstyle)
                ax.plot([0, 1], [0, 1], linestyle=':', color='orange')

                ax.set_xlabel("{}-order AUC-PR".format(p))
                ax.set_ylabel("{}-order AUC-PR".format(p + 1), labelpad=3)

                ax.xaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.tick_params(axis='both', which='major')
                ax.set(adjustable='box-forced', aspect='equal')

        else:
            raise NotImplementedError

    axes[0].set_title("Size 4 prediction",
                      fontweight="bold", pad=10)
    axes[1].set_title("Size 5 prediction",
                      fontweight="bold", x=1.1, pad=10)

    return axes


def plot_dots(neg_type, imb, seed=0):
    # columns: features
    fig, axes = plt.subplots(1, 3, sharex=False, sharey=False)
    # fig.set_size_inches((3*3+3, 3*1))
    fig.set_size_inches((2 * 3 + 2, 2 * 1))

    # Plot
    for hs in [4, 5]:
        axes = _plot_subdots(axes, hs, neg_type, imb, seed)

    # plt.tight_layout()

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plots_path = plots_dir + "dots/neg-{}{}_seed-{}-DOTS-FINAL".format(neg_type, imb, seed)
    fig.savefig(plots_path + '.pdf', bbox_inches='tight', pad_inches=0.03)

    print(plots_path)

    fig.show()


def save_legend(n_cols):
    ax = pylab.gca()

    for gn in gns_display:
        dotstyle = dotstyle_map[gn]
        ax.plot([], label='{}'.format(gn), **dotstyle)
    ax.set_axis_off()

    legend = plt.legend(*ax.get_legend_handles_labels(), loc='center', ncol=n_cols, frameon=False)

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(plots_dir + 'legend/gn_legend_col{}.pdf'.format(n_cols), dpi="figure", bbox_inches=bbox)


if __name__ == "__main__":

    gns_display = ['email-Enron',
                   'email-Eu',
                   'contact-primary-school',
                   'contact-high-school',
                   'NDC-classes',
                   'NDC-substances',
                   'DAWN',
                   'congress-bills',
                   'tags-ask-ubuntu',
                   'tags-math-sx',
                   'threads-ask-ubuntu',
                   'threads-math-sx',
                   'coauth-MAG-History',
                   'coauth-MAG-Geology',
                   'coauth-DBLP']

    linestyle_map = defaultdict(str)
    dotstyle_map = defaultdict(str)
    linestyles = mpltex.linestyle_generator()
    dotstyles = mpltex.linestyle_generator(lines=[])
    for gn in gns_display:
        if gn == "DAWN":  # I don't like pink
            next(linestyles)
            next(dotstyles)
        linestyle_map[gn] = next(linestyles)
        dotstyle_map[gn] = next(dotstyles)

    """ Legnd """
    save_legend(n_cols=3)

    """ Dot plot """
    for neg in ["hub", "clique"]:
        for imb in [10]:
            plot_dots(neg_type=neg, imb=imb, seed=0)

    """ MI Correlation plot """
    for imb in [10, 5, 2, 1]:
        for neg in ["hub"]:
            plot_mi32_correlations(neg_type=neg, imb=imb, seed=0)
            plot_mi42_correlations(neg_type=neg, imb=imb, seed=0)

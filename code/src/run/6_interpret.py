from __future__ import print_function

from collections import defaultdict

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import mpltex
import numpy as np
import pandas as pd
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


def _plot_statistic_correlation(axes, neg_type, imb, seed, row_idx, col_idx, hs, which_stat):
    mean_df = _get_feature_mean_df(hs, neg_type, imb, seed)

    pgs_df = pd.read_csv(statistics_dir + "pgs-stat.csv", index_col=0)

    dots_x = []
    dots_y = []

    for gn in gns_display:

        dotstyle = dotstyle_map[gn]

        pr_row = mean_df.loc[gn]
        pgs_row = pgs_df.loc[gn]

        if which_stat == "percent":
            ax = axes[row_idx][col_idx]
            ax = axes[row_idx][col_idx]

            pr = pr_row.loc["%PR2"]
            percent = pgs_row.loc["3-pg-percent-e"]

            ax.plot(percent,
                    pr,
                    label='{}'.format(gn),
                    **dotstyle)

            ax.tick_params(axis='both', which='major')

            dots_x.append(percent)
            dots_y.append(pr)
        else:
            raise NotImplementedError

    cc = np.corrcoef(dots_x, dots_y)[0][1]

    anc = AnchoredText("CorrCoef: {}".format(round(cc, 2)), loc="lower right", frameon=True)
    ax.add_artist(anc)
    ax.add_artist(anc)

    return axes


def _plot_mi_correlation(axes, neg_type, imb, seed, row_idx, col_idx, hs, which_mi):
    mean_df = _get_feature_mean_df(hs, neg_type, imb, seed)

    mi_df = pd.read_csv(statistics_dir + "mis_exp.csv", index_col=0)

    dots_x = []
    dots_y = []

    for gn in gns_display:

        dotstyle = dotstyle_map[gn]

        pr_row = mean_df.loc[gn]
        mi_row = mi_df.loc[gn]

        if which_mi == "32":
            ax = axes[row_idx][col_idx]

            pr = pr_row.loc["%PR2"]
            mi = mi_row.loc["mi_32"]

            ax.plot(mi,
                    pr,
                    label='{}'.format(gn),
                    **dotstyle)

            ax.tick_params(axis='both', which='major')

            dots_x.append(mi)
            dots_y.append(pr)

        elif which_mi == "42":
            assert (hs == 4)
            ax = axes[row_idx][col_idx]

            pr = pr_row.loc["PR1"]
            mi = mi_row.loc["mi_42"]

            ax.plot(mi,
                    pr,
                    label='{}'.format(gn),
                    **dotstyle)

            ax.tick_params(axis='both', which='major')

            ax.set_xlabel("$I(X_4;X_2)$")
            ax.set_ylabel("2-order AUC-PR", rotation='vertical', labelpad=1)

            dots_x.append(mi)
            dots_y.append(pr)

    cc = np.corrcoef(dots_x, dots_y)[0][1]

    anc = AnchoredText("CorrCoef: {}".format(round(cc, 2)), loc="upper right", frameon=True)
    ax.add_artist(anc)
    ax.add_artist(anc)

    return axes


def _plot_ci_correlation(axes, neg_type, imb, seed, row_idx, col_idx, hs, which_ci):
    mean_df = _get_feature_mean_df(hs, neg_type, imb, seed)

    mi_df = pd.read_csv(statistics_dir + "cis_exp.csv", index_col=0)

    dots_x = []
    dots_y = []

    for gn in gns_display:

        dotstyle = dotstyle_map[gn]

        pr_row = mean_df.loc[gn]
        mi_row = mi_df.loc[gn]

        if which_ci == "32":
            ax = axes[row_idx][col_idx]

            pr = pr_row.loc["%PR2"]
            mi = mi_row.loc["ci_32"]

            ax.plot(mi,
                    pr,
                    label='{}'.format(gn),
                    **dotstyle)

            ax.tick_params(axis='both', which='major')

            dots_x.append(mi)
            dots_y.append(pr)

        elif which_ci == "42":
            assert (hs == 4)
            ax = axes[row_idx][col_idx]

            pr = pr_row.loc["PR1"]
            mi = mi_row.loc["ci_42"]

            ax.plot(mi,
                    pr,
                    label='{}'.format(gn),
                    **dotstyle)

            ax.tick_params(axis='both', which='major')

            dots_x.append(mi)
            dots_y.append(pr)

    cc = np.corrcoef(dots_x, dots_y)[0][1]

    anc = AnchoredText("CorrCoef: {}".format(round(cc, 2)), loc="upper left", frameon=True)
    ax.add_artist(anc)
    ax.add_artist(anc)

    return axes


def _plot_legend(axes, row_idx, col_idx):
    ax = axes[row_idx][col_idx]
    ax.set_axis_off()
    handles, labels = axes[0][1].get_legend_handles_labels()
    ax.legend(handles, labels, loc=(-0.2, -0.15), fontsize=8, ncol=1,
              handletextpad=2, labelspacing=0.4, columnspacing=0.5, frameon=False)
    return axes


def plot_interpretation(neg_type="clique", imb=1, seed=0):
    fig, axes = plt.subplots(2, 3, sharex=False, sharey=False)
    fig.set_size_inches((7, 4), forward=True)

    axes = _plot_statistic_correlation(axes, neg_type, imb, seed, row_idx=0, col_idx=0, hs=4, which_stat="percent")
    axes = _plot_mi_correlation(axes, neg_type, imb, seed, row_idx=0, col_idx=1, hs=4, which_mi="32")
    axes = _plot_ci_correlation(axes, neg_type, imb, seed, row_idx=0, col_idx=2, hs=4, which_ci="32")

    axes = _plot_statistic_correlation(axes, neg_type, imb, seed, row_idx=1, col_idx=0, hs=5, which_stat="percent")
    axes = _plot_mi_correlation(axes, neg_type, imb, seed, row_idx=1, col_idx=1, hs=5, which_mi="32")
    axes = _plot_ci_correlation(axes, neg_type, imb, seed, row_idx=1, col_idx=2, hs=5, which_ci="32")

    axes[1][0].set_xlabel("Edge density in 3-pg", fontweight=None)
    axes[1][1].set_xlabel("$I(W_3;W_2)$")
    axes[1][2].set_xlabel("$H(W_3|W_2)$")

    axes[0][0].set_ylabel("2 to 3 gain (%)", labelpad=4)
    axes[1][0].set_ylabel("2 to 3 gain (%)", labelpad=4)

    axes[0][2].set_ylabel("Size 4", rotation="270", labelpad=16, fontweight="bold")
    axes[1][2].set_ylabel("Size 5", rotation="270", labelpad=16, fontweight="bold")
    axes[0][2].yaxis.set_label_position("right")
    axes[1][2].yaxis.set_label_position("right")

    loc = "right"
    axes[0][0].set_title('A', loc=loc)
    axes[0][1].set_title('B', loc=loc)
    axes[0][2].set_title('C', loc=loc)
    axes[1][0].set_title('D', loc=loc)
    axes[1][1].set_title('E', loc=loc)
    axes[1][2].set_title('F', loc=loc)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plots_path = plots_dir + "mi/neg-{}{}_seed-{}_INTERPRETATION".format(neg_type, imb, seed)
    fig.savefig(plots_path + '.pdf', bbox_inches='tight', pad_inches=0.03)  # dpi=300

    print(plots_path)

    fig.show()


def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


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

    for imb in [1]:
        for neg in ["clique"]:
            plot_interpretation(neg_type=neg, imb=imb, seed=0)

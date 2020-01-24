from __future__ import print_function

import os

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from src.config import *
from src.config import *

params = {'legend.fontsize': 'x-large',
          'axes.titlesize': 14,
          'axes.labelsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 18}
pylab.rcParams.update(params)


def create_agg_df(hs):
    if hs == 4:
        columns = ["PR1", "PR2", "ROC1", "ROC2", "/", "%PR2", "%ROC2"]
    else:
        columns = ["PR1", "PR2", "PR3", "ROC1", "ROC2", "ROC3", "/", "%PR2", "%PR3", "%ROC2", "%PR3"]

    rows = ['email-Enron',
            'contact-primary-school',
            'contact-high-school',
            'NDC-classes',
            'NDC-substances',
            'email-Eu',
            'threads-ask-ubuntu',
            'congress-bills',
            'DAWN',
            'tags-ask-ubuntu',
            'tags-math-sx',
            'threads-math-sx',
            'coauth-MAG-History',
            'coauth-MAG-Geology',
            'coauth-DBLP',
            ]

    df_agg = pd.DataFrame(index=rows, columns=columns)

    return df_agg


def get_agg_df(feature, hs, imb, neg_type, seed=0):
    from src.config import results_dir

    # agg dir
    results_dir = results_dir.format(hs, neg_type, imb, seed)
    agg_dir = results_dir + 'agg/'
    agg_path = agg_dir + '{}-aggregate.csv'.format(feature)
    agg_df = pd.read_csv(agg_path, index_col=0)

    return agg_df


def _subgrid(axes, seed, feature, hs):
    hs_idx = hss.index(hs)
    f_idx = features.index(feature)

    ax = axes[hs_idx][f_idx]

    neg_types = ["hub", "clique"]
    imbs = [1, 2, 5, 10]

    grid_df = pd.DataFrame(index=neg_types, columns=imbs)

    for x, imb in enumerate(imbs):
        for y, neg_type in enumerate(neg_types):
            agg_df = get_agg_df(feature, hs, imb, neg_type, seed)

            inc2 = agg_df.loc[:, "%PR2"]
            inc2_mean = inc2.dropna().mean()
            grid_df.loc[neg_type, imb] = inc2_mean

            text = ax.text(x, y, inc2_mean.round(1), ha="center", va="center", color="w")
            plt.draw()

    grid = grid_df.as_matrix()
    grid = np.asarray(grid, dtype='float')

    if hs == 4:
        color = 'Blues'
    elif hs == 5:
        color = 'Oranges'
        # color = 'Greens'
    else:
        color = 'Oranges'

    cnap = cm.get_cmap(color, 20)
    cmap = ListedColormap(cnap(np.linspace(0.25, 1, 20)))

    im = ax.imshow(grid, cmap, origin="upper")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(imbs)))
    ax.set_yticks(np.arange(len(neg_types)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(["1:{}".format(imb) for imb in imbs])
    ax.set_yticklabels(["", ""])

    axes[-1][f_idx].set_xlabel("Class imbalance".format(imb))

    axes[0][f_idx].set_title(["GM", "HM", "AM", "CN", "JC", "AA"][f_idx], fontweight="bold")
    axes[hs_idx][-1].set_ylabel("Size {}".format(hs), rotation=270, labelpad=20, fontweight="bold")
    ax.yaxis.set_label_position("right")

    return axes, im


# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def grid(seed=0):
    from src.config import grids_dir

    # rows: hedge sizes
    # columns: features
    fig, axes = plt.subplots(len(hss), len(features), sharex=True, sharey=True)
    fig.set_size_inches((14.5, 3.2), forward=True)

    # Plot
    for feature in features:
        for hs in hss:
            axes, im = _subgrid(axes, seed, feature=feature, hs=hs)

    if not os.path.exists(grids_dir):
        os.makedirs(grids_dir)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    grids_path = grids_dir + "heatmap".format(seed)
    fig.savefig(grids_path + '.pdf', bbox_layout="tight", pad_inches=0.01)

    print(grids_path)

    fig.show()


def draw_subgraph(subn, G):
    subg = G.subgraph(subn)
    nx.draw_networkx(subg,
                     with_labels=False,
                     node_size=100,
                     node_color='b')

    nx.draw_networkx_edge_labels(subg, pos=nx.spring_layout(subg))
    plt.show()


def scsc():
    node_size = 50
    node_color = 'mediumpurple'

    fig, axes = plt.subplots(4, 1)
    fig.set_size_inches((2, 8))

    for i in range(4):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_axis_off()

    star = nx.star_graph(n=3)
    nx.draw_networkx(star,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[0])

    clique = nx.complete_graph(n=4)
    nx.draw_networkx(clique,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[1])

    star = nx.star_graph(n=4)
    nx.draw_networkx(star,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[2])

    clique = nx.complete_graph(n=5)
    nx.draw_networkx(clique,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[3])

    plt.show()

    fig.savefig(grids_dir + 'scsc.pdf', bbox_inches='tight', pad_inches=0.03)


def sscc():
    node_size = 50
    node_color = 'mediumpurple'

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches((4, 4))

    for i in [0, 1]:
        for j in [0, 1]:
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            axes[i][j].set_axis_off()

    star = nx.star_graph(n=3)
    nx.draw_networkx(star,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[0][0])

    clique = nx.complete_graph(n=4)
    nx.draw_networkx(clique,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[1][0])

    star = nx.star_graph(n=4)
    nx.draw_networkx(star,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[0][1])

    clique = nx.complete_graph(n=5)
    nx.draw_networkx(clique,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[1][1])

    plt.show()

    fig.savefig(grids_dir + 'sscc.pdf', bbox_inches='tight', pad_inches=0.0)


def ss_cc():
    node_size = 50
    node_color = 'mediumpurple'

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((4, 2))

    for i in [0, 1]:
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_axis_off()

    star = nx.star_graph(n=3)
    nx.draw_networkx(star,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[0])

    star = nx.star_graph(n=4)
    nx.draw_networkx(star,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[1])

    fig.savefig(grids_dir + 'ss.pdf', bbox_inches='tight', pad_inches=0.0)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((4, 2))

    for i in [0, 1]:
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_axis_off()

    clique = nx.complete_graph(n=4)
    nx.draw_networkx(clique,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[0])

    clique = nx.complete_graph(n=5)
    nx.draw_networkx(clique,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color,
                     ax=axes[1])

    fig.savefig(grids_dir + 'cc.pdf', bbox_inches='tight', pad_inches=0.0)


if __name__ == "__main__":
    args = parse_args()

    grid()

    # scsc()
    #
    # sscc()
    #
    # ss_cc()

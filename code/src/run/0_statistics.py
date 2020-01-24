from __future__ import print_function

import operator as op
import os
from functools import reduce
from itertools import combinations_with_replacement
from math import log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from src.config import *
from src.lib.utils import *


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def create_pg_df():
    columns = []
    for p in range(2, 10):
        columns.append("{}-pg #n".format(p))
        columns.append("{}-pg #e".format(p))

    rows = [
        'email-Enron',
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
        'coauth-DBLP'
    ]

    agg_df = pd.DataFrame(index=rows, columns=columns)

    return agg_df


def pg_statistics(gn):
    pg_dir = statistics_dir
    if not os.path.exists(pg_dir):
        os.makedirs(pg_dir)

    # For recording performance in an aggregated file
    pg_dir = pg_dir + 'pgs-new.csv'
    if os.path.isfile(pg_dir):
        pg_df = pd.read_csv(pg_dir, index_col=0)
    else:
        pg_df = create_pg_df()

    transactions, hedges, weights = get_transactions_hedges_and_weights(gn)

    for p in range(5, 10):
        pg = hedges_to_pg(hedges, weights, p=p)

        pg_df.loc[gn, "{}-pg #n".format(p)] = int(pg.number_of_nodes())
        pg_df.loc[gn, "{}-pg #e".format(p)] = int(pg.number_of_edges())

    # aggregate results
    pg_df.to_csv(pg_dir)


def plot_hist(gn):
    BINS = 100

    transactions, hedges, weights = get_transactions_hedges_and_weights(gn)

    # figs
    n_axes = 4
    fig, axes = plt.subplots(n_axes, 1, sharex=True, sharey=True)
    fig.set_size_inches((7, 2 * n_axes))
    ax = plt.gca()

    for idx, set_size in enumerate([1, 2, 3, 4]):
        color = next(ax._get_lines.prop_cycler)['color']

        supcount = hedges_to_supcount(hedges, weights, set_size=set_size)

        values = supcount.values()
        values = [v for v in values if v < 500]

        axes[idx].set_xlabel('Support count')
        axes[idx].set_ylabel('# size {} sets'.format(set_size))
        axes[idx].hist(values, bins=BINS, color=color)

        plt.yscale('symlog')

    # save fig
    dir = statistics_dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir + '{}-hist'.format(gn)

    fig.suptitle(gn, y=0.95)

    fig.savefig(path)
    # plt.show()


def _interval_idx(val):
    """
    round(log(val+1, 2))

    0: 0
    1: 1
    2~4: 2
    ...
    ~361: 8
    362~: 9
    """
    if val >= 362:
        idx = 9
    else:
        idx = round(log(val + 1, 2))

    return idx


def create_info_df(which="mi"):
    columns = ["{}_32".format(which), "{}_43".format(which), "{}_42".format(which)]

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


def get_joint_arr(hsc, sc, hx, x):
    n_combinations = ncr(hx, x)

    tups = [sorted(combination) for combination in combinations_with_replacement(range(0, 11), n_combinations)]

    joint_df = pd.DataFrame(0,
                            index=pd.MultiIndex.from_tuples(tups),
                            columns=range(0, 11))

    # print(joint_df)

    for hset in hsc:

        hset_val = hsc[hset]
        hset_idx = _interval_idx(hset_val)

        set_idxs = []
        for combination in combinations(hset, x):
            set_val = sc[combination]
            set_idx = _interval_idx(set_val)

            set_idxs.append(set_idx)

        set_idxs = tuple(sorted(set_idxs))

        joint_df.loc[set_idxs, hset_idx] += 1

    # print(joint_df)
    joint_arr = joint_df.as_matrix()

    return joint_arr


def marginal_from_joint(joint_arr, axis=1):
    marginal_arr = np.sum(joint_arr, axis)

    print(joint_arr.shape)
    print(marginal_arr.shape)

    return marginal_arr


def plot_joint_arr(hsc, sc, hx, x):
    for hset in hsc:

        hset_val = hsc[hset]

        set_vals = []
        for combination in combinations(hset, x):
            set_val = sc[combination]
            set_vals.append(set_val)
        temp = np.mean(set_vals)

        plt.plot(temp, hset_val, 'o')

    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def mutual_information(hsc, sc, hx, x):
    """
    I(hx;x) = H(hx) - H(hx|x)
    """
    joint_arr = get_joint_arr(hsc, sc, hx, x)
    mi = mutual_info_score(None, None, contingency=joint_arr)

    return mi


def conditional_information(hsc, sc, hx, x):
    """
    H(hx|x) = H(hx) - I(hx;x)
    """
    joint_arr = get_joint_arr(hsc, sc, hx, x)
    marginal_arr = marginal_from_joint(joint_arr)

    mi = mutual_info_score(None, None, contingency=joint_arr)

    ci = entropy(marginal_arr) - mi

    return ci


def mi_statistics(gns):
    mi_dir = statistics_dir
    if not os.path.exists(mi_dir):
        os.makedirs(mi_dir)

    for gn in gns:
        print('\n')
        print(gn)
        transactions, hedges, weights = get_transactions_hedges_and_weights(gn)

        for hx, x in [(3, 2), (4, 3), (4, 2)]:

            hsc = hedges_to_supcount(hedges, weights, set_size=hx)
            sc = hedges_to_supcount(hedges, weights, set_size=x)

            mi = mutual_information(hsc, sc, hx, x)

            # For recording performance in an aggregated file
            mi_path = mi_dir + 'mis_exp.csv'
            if os.path.isfile(mi_path):
                mi_df = pd.read_csv(mi_path, index_col=0)
            else:
                mi_df = create_info_df(which="mi")
            mi_df.loc[gn, "mi_{}{}".format(hx, x)] = mi

            mi_df.to_csv(mi_path)

        print(mi_df)


def ci_statistics(gns):
    mi_dir = statistics_dir
    if not os.path.exists(mi_dir):
        os.makedirs(mi_dir)

    for gn in gns:
        print('\n')
        print(gn)
        transactions, hedges, weights = get_transactions_hedges_and_weights(gn)

        for hx, x in [(3, 2), (4, 3), (4, 2)]:

            hsc = hedges_to_supcount(hedges, weights, set_size=hx)
            sc = hedges_to_supcount(hedges, weights, set_size=x)

            ci = conditional_information(hsc, sc, hx, x)

            # For recording performance in an aggregated file
            ci_path = mi_dir + 'cis_exp.csv'
            if os.path.isfile(ci_path):
                ci_df = pd.read_csv(ci_path, index_col=0)
            else:
                ci_df = create_info_df(which="ci")
            ci_df.loc[gn, "ci_{}{}".format(hx, x)] = ci

            ci_df.to_csv(ci_path)

        print(ci_df)
        print(ci_path)

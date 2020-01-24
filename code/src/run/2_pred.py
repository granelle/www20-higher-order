from __future__ import print_function

import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from src.config import *
from src.config import *
from src.lib.dataset import HedgePredDataset
from src.lib.features import get_feature_vectors_dict

ax = plt.gca()


def train_test_logreg(dataset_train, dataset_test, df, dim):
    # Train dataset
    inputs_train = dataset_train.x_data  # 2d tensor
    labels_train = dataset_train.y_data

    # Test dataset
    inputs_test = dataset_test.x_data  # 2d tensor
    labels_test = dataset_test.y_data

    # Train
    clf = LogisticRegression(random_state=0,
                             solver='lbfgs',
                             multi_class='ovr',
                             max_iter=100000)
    clf.fit(np.array(inputs_train), np.ravel(labels_train))

    # Test
    # y_pred = clf.predict(inputs_test)
    y_pred = clf.predict_proba(inputs_test)
    y_pred = y_pred[:, 1]

    y_true = labels_test

    # df: inputs
    inputs = inputs_test.numpy()
    inputs = inputs[:, dim - 1]
    df['input_dim{}'.format(dim)] = pd.Series(inputs)

    # df: predictions
    preds = clf.predict(inputs_test)
    df['pred_dim{}'.format(dim)] = pd.Series(preds)

    # df: labels
    labels = labels_train.numpy()
    labels = labels[:, 0]
    df['label'] = pd.Series(labels)

    return y_true, y_pred, df


def evaluate_performance(y_true, y_pred):
    # precision, recall, fpr, tpr
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    auc_pr = auc(recall, precision)
    auc_roc = auc(fpr, tpr)

    return precision, recall, fpr, tpr, auc_pr, auc_roc


def record_performance(y_true, y_pred):
    raise NotImplementedError


def draw_curves(y_true, y_pred, features, dim, axes):
    precision, recall, fpr, tpr, auc_pr, auc_roc = evaluate_performance(y_true, y_pred)

    print("-----------------------")
    print("auc_pr for {}-{}: {}".format(features, dim, auc_pr))
    print("auc_roc for {}-{}: {}".format(features, dim, auc_roc))

    color = next(ax._get_lines.prop_cycler)['color']

    linestyles = [':', '--', '-']

    linestyle = linestyles[dim - 1]

    axes[0].plot(recall, precision, linestyle=linestyle, color=color,
                 label='{}-{} (test) val: {:10.4f}'.format(features, dim, auc_pr))
    axes[0].legend(loc='best')

    axes[1].plot(fpr, tpr, linestyle=linestyle, color=color,
                 label='{}-{} (test) val: {:10.4f}'.format(features, dim, auc_roc))
    axes[1].legend(loc='best')

    return axes


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


def insert_agg_df(df_agg, gn, dim, auc_pr, auc_roc):
    df_agg.loc[gn, "PR{}".format(dim)] = auc_pr
    df_agg.loc[gn, "ROC{}".format(dim)] = auc_roc

    if dim > 1:
        pr_prev = df_agg.loc[gn, "PR{}".format(dim - 1)]
        roc_prev = df_agg.loc[gn, "ROC{}".format(dim - 1)]

        ppr = (auc_pr - pr_prev) / float(pr_prev) * 100
        proc = (auc_roc - roc_prev) / float(roc_prev) * 100

        df_agg.loc[gn, "%PR{}".format(dim)] = ppr
        df_agg.loc[gn, "%ROC{}".format(dim)] = proc

    return df_agg


def hedgepred(hs, neg_type, imb, seed, features, gn):
    from src.config import hg_dir, results_dir

    print("hs: {}".format(hs))

    print("neg_type: {}".format(neg_type))

    print("imb: {}".format(imb))

    print("seed: {}".format(seed))

    print("gn: {}".format(gn))

    print("\n")

    # Cached data
    if int(imb) < 10:  # Load from 10 but use just a fraction of it
        hg_dir = hg_dir.format(gn, hs, neg_type, "10", seed)
    else:
        hg_dir = hg_dir.format(gn, hs, neg_type, imb, seed)

    print("Loading cached data..")
    start_time = time.time()
    with open(hg_dir + 'hg.pos', 'rb') as f:
        pos = pickle.load(f)
    with open(hg_dir + 'hg.neg', 'rb') as f:
        neg = pickle.load(f)
    with open(hg_dir + 'hg.pgs', 'rb') as f:
        pgs = pickle.load(f)
    print("..done: {} seconds\n".format(time.time() - start_time))

    # Extract just a fraction of negative samples
    if int(imb) < 10:
        random.seed(seed)
        neg_train = random.sample(neg["train"], k=int(len(neg["train"]) * float(imb) / 10.0))
        neg_test = random.sample(neg["test"], k=int(len(neg["test"]) * float(imb) / 10.0))
    else:
        neg_train = neg["train"]
        neg_test = neg["test"]

    # Results
    results_dir = results_dir.format(hs, neg_type, imb, seed)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # figs
    n_axes = 2
    fig, axes = plt.subplots(n_axes, 1, sharex=True, sharey=True)
    fig.set_size_inches((8, 8 * n_axes))

    # axes
    axes[0].set_title('AUC-PR')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')

    axes[1].set_title('AUC-ROC')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='y')

    # df
    df = pd.DataFrame(
        columns=['input_dim1', 'pred_dim1', 'input_dim2', 'pred_dim2', 'input_dim3', 'pred_dim3', 'label'])

    if hs == 4:
        dims = [1, 2]
    elif hs > 4:
        dims = [1, 2, 3]
    else:
        raise NotImplementedError

    for dim in [3]:  # for dim in dims:  # Todo: this is only temporary

        print("Preparing {}-dim dataset..".format(dim))
        start_time = time.time()

        pos_feats_dict_train = get_feature_vectors_dict(hedges=pos["train"],
                                                        pg2=pgs["pg2_train"],
                                                        pg3=pgs["pg3_train"],
                                                        pg4=pgs["pg4_train"],
                                                        dim=dim)

        neg_feats_dict_train = get_feature_vectors_dict(hedges=neg_train,
                                                        pg2=pgs["pg2_train"],
                                                        pg3=pgs["pg3_train"],
                                                        pg4=pgs["pg4_train"],
                                                        dim=dim)

        pos_feats_dict_test = get_feature_vectors_dict(hedges=pos["test"],
                                                       pg2=pgs["pg2_test"],
                                                       pg3=pgs["pg3_test"],
                                                       pg4=pgs["pg4_test"],
                                                       dim=dim)

        neg_feats__dict_test = get_feature_vectors_dict(hedges=neg_test,
                                                        pg2=pgs["pg2_test"],
                                                        pg3=pgs["pg3_test"],
                                                        pg4=pgs["pg4_test"],
                                                        dim=dim)

        print("..done: {} seconds\n".format(time.time() - start_time))

        for feature in features:

            # Aggregate results
            agg_dir = results_dir + 'agg/'
            if not os.path.exists(agg_dir):
                os.makedirs(agg_dir)

            # For recording performance in an aggregated file
            agg_path = agg_dir + '{}-aggregate.csv'.format(feature)
            if os.path.isfile(agg_path):
                agg_df = pd.read_csv(agg_path, index_col=0)
            else:
                agg_df = create_agg_df(hs)

            print("Train/testing..")

            print("feature: {}".format(feature))

            dataset_train = HedgePredDataset(pos_feats=pos_feats_dict_train[feature],
                                             neg_feats=neg_feats_dict_train[feature])
            dataset_test = HedgePredDataset(pos_feats=pos_feats_dict_test[feature],
                                            neg_feats=neg_feats__dict_test[feature])

            # y_true, y_pred
            y_true, y_pred, df = train_test_logreg(dataset_train, dataset_test, df, dim)

            # performance
            precision, recall, fpr, tpr, auc_pr, auc_roc = evaluate_performance(y_true, y_pred)

            # record
            # record_performance(y_true, y_pred)

            # plot
            axes = draw_curves(y_true, y_pred, feature, dim, axes)

            # insert
            agg_df = insert_agg_df(agg_df, gn, dim, auc_pr, auc_roc)

            # aggregate results
            agg_df.to_csv(agg_path)
            print(agg_df)
            print(agg_path)

            # individual results
            results_path = results_dir + "{}-{}".format(feature, gn)
            # fig.suptitle(results_path)
            # fig.savefig(results_path + '.png')
            df.to_csv(results_path + '.csv', mode='w')
            print(results_path)


if __name__ == "__main__":

    start_time = time.time()

    print("Prediction start\n")

    args = parse_args()

    hs = 5

    neg_type = "clique"

    # imb = 5

    seed = args.seed

    gn = 'tags-ask-ubuntu'

    for imb in [10]:
        hedgepred(hs, neg_type, imb, seed, features, gn)

    print("--- {} seconds ---\n".format(time.time() - start_time))

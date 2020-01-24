from __future__ import print_function

import pandas as pd
from src.config import *
from src.config import *


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


def get_feature_series(feature, hs, dim, imb, neg_type, seed=0):
    from src.config import results_dir

    # agg dir
    results_dir = results_dir.format(hs, neg_type, imb, seed)
    agg_dir = results_dir + 'agg/'
    agg_path = agg_dir + '{}-aggregate.csv'.format(feature)
    agg_df = pd.read_csv(agg_path, index_col=0)

    # series
    series = []
    for gn in gns_display:
        inc1 = agg_df.loc[gn, "%PR{}".format(dim)]
        series.append(agg_df.loc[gn, "%PR{}".format(dim)])

    return series


def feature_agg_df(hs, imb, neg_type, seed=0):
    from src.config import tables_dir

    # New empty df
    columns = features
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
                   'coauth-DBLP'
                   ]

    cat_agg_df = pd.DataFrame(index=gns_display, columns=columns)

    if hs == 4:
        dims = [2]
    elif hs == 5:
        dims = [2, 3]
    else:
        raise NotImplementedError

    for dim in dims:
        for feature in features:
            series = get_feature_series(feature=feature, hs=hs, dim=dim, imb=imb, neg_type=neg_type, seed=seed)

            cat_agg_df[feature] = series

        tables_path = tables_dir + "size-{}_dim-{}_neg-{}{}_seed-{}-TABLE".format(hs, dim, neg_type, imb, seed)
        print(tables_path)

        cat_agg_df.to_csv(tables_path + '.csv', mode='w')
        print(cat_agg_df)


if __name__ == "__main__":

    # gns = gns
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
    features = features

    for neg in ["hub", "clique"]:
        for hs in [4, 5]:
            feature_agg_df(hs=hs, imb=10, neg_type=neg)

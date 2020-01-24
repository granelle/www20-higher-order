from itertools import combinations
from math import log

import numpy as np
from src.lib.utils import *


def _get_neighbors(pg, node):
    if node in pg:
        neighbors = set(pg.neighbors(node))
    else:
        neighbors = set([])

    return neighbors


def neighbors(hedge, pg, p):
    assert (p in [2, 3, 4])

    nodes = hedge_to_pg_nodes(hedge, p)

    first = True
    for node in nodes:

        if first:
            prev_n = _get_neighbors(pg, node)
            first = False

        neighbors = _get_neighbors(pg, node)
        cns = prev_n.intersection(neighbors)
        uns = prev_n.union(neighbors)

    cn = len(cns)

    if len(uns) == 0:
        jc = 0
    else:
        jc = len(cns) / float(len(uns))

    aa = 0
    for node in cns:
        aa += 1 / float(log(len(_get_neighbors(pg, node))))

    return cn, jc, aa


def means(hedge, pg, p):  # note that this calculates "without duplicate"

    assert (p in [2, 3, 4])

    w_sum = 0  # arithmetic
    w_mul = 1  # geometric
    w_invsum = 0  # harmonic

    N = 0

    for combination in combinations(hedge, p):

        if p == 2:
            node1 = combination[0]
            node2 = combination[1]

        if p == 3:
            node1 = (combination[0], combination[1])
            node2 = (combination[0], combination[2])

        if p == 4:
            node1 = (combination[0], combination[1], combination[3])
            node2 = (combination[0], combination[2], combination[3])

        if pg.has_edge(node1, node2):
            w = pg[node1][node2]['weight']
        else:
            w = 0

        w_sum += w  # arithmetic
        w_mul *= w  # geometric
        if w != 0:
            w_invsum += 1 / float(w)  # harmonic
        N += 1

    am = w_sum / float(N)
    gm = w_mul ** (1 / float(N))
    if w_invsum != 0:
        hm = float(N) / w_invsum
    else:
        hm = 0

    return gm, hm, am


def _append_features(vector_dict, hedge, pg, p):
    gm, hm, am = means(hedge, pg, p)

    vector_dict["gm"].append(gm)
    vector_dict["hm"].append(hm)
    vector_dict["am"].append(am)

    cn, jc, aa = neighbors(hedge, pg, p)

    vector_dict["cn"].append(cn)
    vector_dict["jc"].append(jc)
    vector_dict["aa"].append(aa)

    return vector_dict


def get_feature_vector_dict(hedge, pg2, pg3, pg4, dim):
    vector_dict = {"gm": [],
                   "hm": [],
                   "am": [],
                   "cn": [],
                   "jc": [],
                   "aa": []}

    if dim > 0:
        vector_dict = _append_features(vector_dict, hedge, pg2, p=2)

    if dim > 1:
        vector_dict = _append_features(vector_dict, hedge, pg3, p=3)

    if dim > 2:
        vector_dict = _append_features(vector_dict, hedge, pg4, p=4)

    for key in vector_dict:
        vector_dict[key] = np.asarray(vector_dict[key])

    return vector_dict


def get_feature_vectors_dict(hedges, pg2, pg3, pg4, dim):
    vectors_dict = {"gm": [],
                    "hm": [],
                    "am": [],
                    "cn": [],
                    "jc": [],
                    "aa": []}

    for hedge in hedges:
        for key in vectors_dict:
            vector = get_feature_vector_dict(hedge, pg2, pg3, pg4, dim)[key]
            vectors_dict[key].append(vector)

    for key in vectors_dict:
        vectors_dict[key] = np.asarray(vectors_dict[key])

    return vectors_dict


if __name__ == "__main__":
    gn = 'email-Enron'
    # gn = 'coauth-DBLP'

    transactions, hedges, weights = get_transactions_hedges_and_weights(gn)

    pg2 = hedges_to_pg(hedges, weights, p=2)
    pg3 = hedges_to_pg(hedges, weights, p=3)
    pg4 = hedges_to_pg(hedges, weights, p=4)

    hedges = [(18, 84, 106, 109, 130, 144)]

    hedge = hedges[0]

    gm2, hm2, am2 = means(hedge, pg2, p=2)
    gm3, hm3, am3 = means(hedge, pg3, p=3)
    gm4, hm4, am4 = means(hedge, pg4, p=4)

    cn2, jc2, aa2 = neighbors(hedge, pg2, p=2)
    cn3, jc3, aa3 = neighbors(hedge, pg3, p=3)
    cn4, jc4, aa4 = neighbors(hedge, pg4, p=4)

    print(aa2, cn2, am2, hm2, gm2, jc2)
    vector_dict = get_feature_vector_dict(hedges[0], pg2, pg3, pg4, dim=1)
    print(vector_dict)
    print("---------------------")

    print(aa3, cn3, am3, hm3, gm3, jc3)
    vector_dict = get_feature_vector_dict(hedges[0], pg2, pg3, pg4, dim=2)
    print(vector_dict)
    print("---------------------")

    print(aa4, cn4, am4, hm4, gm4, jc4)
    vector_dict = get_feature_vector_dict(hedges[0], pg2, pg3, pg4, dim=3)
    print(vector_dict)
    print("---------------------")

    print("\n\n")

    vectors_dict = get_feature_vectors_dict(hedges, pg2, pg3, pg4, dim=2)

    print(vectors_dict)

from itertools import combinations

import networkx as nx
from src.config import *


def get_n_nodes(gn):
    with open(dataset_dir + '{}/{}-node-labels.txt'.format(gn, gn), 'r') as f:
        for i, line in enumerate(f):
            pass
    return i + 1


def get_transactions_hedges_and_weights(gn):
    timestamped_hedges = []
    unique_hedges = []
    weights = []

    # Generate timestamped hedges
    print("Generating timestamped hedges...")
    with open(dataset_dir + '{}/{}-simplices.txt'.format(gn, gn), 'r') as f_simplices:
        with open(dataset_dir + '{}/{}-nverts.txt'.format(gn, gn), 'r') as f_nverts:
            for nverts in f_nverts:
                # hedge
                hedge = []
                for i in range(int(nverts)):
                    line = f_simplices.readline()
                    hedge.append(int(line))
                if int(nverts) > MAX_HEDGE_SIZE:  # skip this simplex
                    continue
                hedge.sort()
                timestamped_hedges.append(tuple(hedge))
    print("...done")

    # Sort hedges
    hedges = sorted(timestamped_hedges)

    # Generate unique hedges and their weights
    print("Generating unique hedges...")
    for idx, hedge in enumerate(hedges):
        if idx == 0:
            last_hedge = hedge
            w = 1
            continue
        if hedge == last_hedge:
            w += 1
            continue
        unique_hedges.append(last_hedge)
        weights.append(w)
        last_hedge = hedge
        w = 1
    # last element
    unique_hedges.append(last_hedge)
    weights.append(w)
    print("...done")

    return timestamped_hedges, unique_hedges, weights


def cache_transactions_hedges_and_weights(timestamped_hedges, unique_hedges, weights):
    cached_hedges = {}

    cached_hedges['timestamped'] = timestamped_hedges
    cached_hedges['unique'] = unique_hedges
    cached_hedges['weights'] = weights

    return cached_hedges


def hedges_to_pg(hedges, weights, p):
    print("Generating {}-pg..".format(p))

    assert (p >= 2)

    pg = nx.Graph()

    node_size = p - 1

    n_edges = 0
    for idx, hedge in enumerate(hedges):

        if len(hedge) < p:
            continue

        w = weights[idx]

        if p == 2:
            nodes = hedge
        else:
            nodes = []
            for node in combinations(hedge, node_size):
                nodes.append(node)

        # Update edges
        for edge in combinations(nodes, 2):
            node1, node2 = edge

            if p > 2:
                if len(set(node1 + node2)) > p:
                    continue

            if pg.has_edge(node1, node2):
                pg[node1][node2]['weight'] += w
            else:
                pg.add_edge(node1, node2, weight=w)
                n_edges += 1
                if n_edges % (500 * 1000) == 0:
                    print(n_edges)
                if n_edges == 500 * 1000 * 1000:  # cannot handle more
                    print("Cannot handle more")
                    return pg

    print("..done")
    return pg


def hedges_to_supcount(hedges, weights, set_size):
    print("Getting supcounts..")

    supcount_dict = {}

    for idx, hedge in enumerate(hedges):

        w = weights[idx]

        for itemset in combinations(hedge, set_size):

            if itemset in supcount_dict:
                supcount_dict[itemset] += w
            else:
                supcount_dict[itemset] = w

    print("..done")
    return supcount_dict


def hedge_to_pg_nodes(hedge, p):
    nodes = []

    if p == 2:
        nodes = list(hedge)
    else:
        for combination in combinations(hedge, p - 1):
            nodes.append(combination)

    return nodes

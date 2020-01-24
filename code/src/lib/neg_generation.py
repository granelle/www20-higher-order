import random

import matplotlib.pyplot as plt
import networkx as nx
from src.config import *


def get_cliques(size, G, exclude, n, strong=False, max_iter=MAX_ITER):
    """
    :param size: clique size
    :param G: nx graph
    :param exclude: exclude these
    :param n: try to find n cliques
    :param strong: strong clique
    :param max_iter: try until this iteration of while loop
    :return:
    """

    cliques = set()

    n_neighbors = size - 1
    n_edges = size * (size - 1) / 2

    nodes = list(G.nodes)

    n_iter = 0
    while len(cliques) < n:
        if len(cliques) >= n:
            break

        if n_iter >= max_iter:
            break
        if n_iter % (10 * 1000) == 0:
            print("n_iter: {}".format(n_iter))

        n_iter += 1

        node = random.choice(nodes)  # select a random node

        all_neighbors = [neigh for neigh in G.neighbors(node)]
        if len(all_neighbors) < n_neighbors:
            continue

        # find at most 1 clique
        for i in range(10):

            neighbors = random.sample(all_neighbors, k=n_neighbors)

            subn = [node] + neighbors
            subg = G.subgraph(subn)

            if subg.number_of_edges() != n_edges:
                continue  # not a clique

            clique = subn

            clique.sort()
            clique = tuple(clique)

            if clique not in exclude:
                cliques.add(clique)
                break

    cliques = list(cliques)  # list of non-overlapping cliques

    print("# node iterations {}".format(n_iter))
    print("# cliques found: {}".format(len(cliques)))

    return cliques


def get_hubs(size, G, exclude, n, strong=False, max_iter=MAX_ITER):
    hubs = set()

    n_neighbors = size - 1

    nodes = list(G.nodes)

    n_iter = 0
    while len(hubs) < n:
        if len(hubs) >= n:
            break

        if n_iter > max_iter:
            break

        n_iter += 1

        node = random.choice(nodes)

        all_neighbors = [neighbor for neighbor in G.neighbors(node)]
        if len(all_neighbors) < n_neighbors:
            continue

        for i in range(min(20, G.degree[node] / 4)):

            neighbors = random.sample(all_neighbors, k=n_neighbors)
            hub = [node] + neighbors

            hub.sort()
            hub = tuple(hub)

            if hub not in exclude:
                hubs.add(hub)
                if len(hubs) >= n:
                    break

    hubs = list(hubs)

    print("# node iterations {}".format(n_iter))
    print("# hubs found: {}".format(len(hubs)))

    return hubs


def get_n_pos(n_all, n_candidates, max_delete_prop=0.4):
    max_pos = int(n_all * max_delete_prop)
    n_pos = min(max_pos, n_candidates)

    return n_pos


def get_n_neg(n_pos, class_imb, max_absolute=MAX_NEG_GENERATION):
    max_neg = int(n_pos * class_imb)
    n_neg = min(max_neg, max_absolute)
    return n_neg


def draw_subgraph(subn, G):
    subg = G.subgraph(subn)
    nx.draw_networkx(subg,
                     with_labels=False,
                     node_size=100,
                     node_color='b')

    nx.draw_networkx_edge_labels(subg, pos=nx.spring_layout(subg))
    plt.show()

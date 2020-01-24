from __future__ import print_function

import time

from src.config import *
from src.lib.hypergraph import HyperGraph
from src.lib.utils import *


def generate_hg(gns, hedge_sizes, neg_types, imbs, seeds):
    print("Generating hg\n")

    for gn in gns:
        print("gn: {}".format(gn))
        hg = HyperGraph(gn=gn)
        hg.read_hedges()

        process_cache_hg(hg, hedge_sizes=hedge_sizes, neg_types=neg_types, imbs=imbs, seeds=seeds)


def process_cache_hg(hg, hedge_sizes, neg_types, imbs, seeds):
    for hedge_size in hedge_sizes:
        for neg_type in neg_types:
            for imb in imbs:
                for seed in seeds:
                    print("hs: {}".format(hedge_size))
                    print("neg_type: {}".format(neg_type))
                    print("imb: {}".format(imb))
                    print("seed: {}".format(seed))

                    start_time = time.time()

                    hg.generate_pos_neg_hedges(hedge_size, neg_type, imb, seed)

                    hg.generate_pgs()

                    hg.cache_hg()

                    print("--- {} seconds ---\n".format(time.time() - start_time))


if __name__ == "__main__":
    gns = gns
    hedge_sizes = [4]

    neg_types = ["clique"]
    imbs = [10]
    seeds = [0]

    generate_hg(gns, hedge_sizes, neg_types, imbs, seeds)

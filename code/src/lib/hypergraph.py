from __future__ import print_function

import os
import pickle
import sys
from collections import defaultdict

from src.config import *
from src.lib.neg_generation import *
from src.lib.utils import *


class HyperGraph:
    def __init__(self, gn):

        self.gn = gn
        self.hedge_size = None
        self.neg_type = None
        self.imb = None
        self.seed = None

        self.transactions = []
        self.hedges = []
        self.weights = []

        # Train
        self.pos_hedges_train = []
        self.neg_hedges_train = []
        self.vis_hedges_train = []
        self.vis_weights_train = []

        self.pg2_train = None
        self.pg3_train = None
        self.pg4_train = None

        # Test
        self.pos_hedges_test = []
        self.neg_hedges_test = []
        self.vis_hedges_test = []
        self.vis_weights_test = []

        self.pg2_test = None
        self.pg3_test = None
        self.pg4_test = None

    def read_hedges(self):

        gn = self.gn

        transactions, hedges, weights = get_transactions_hedges_and_weights(gn)

        self.transactions = transactions
        self.hedges = hedges
        self.weights = weights

    def generate_pos_neg_hedges(self, hedge_size, neg_type, imb, seed):

        load_from_cache = True

        if load_from_cache:
            print("Loading pos neg from cache...")
            hedges = self.hedges
            weights = self.weights

            hg_dir = '../../hg-MAX10/{}/size-{}/neg-{}{}/seed-{}/'.format(self.gn, hedge_size, neg_type, imb, seed)
            with open(hg_dir + 'hg.pos', 'rb') as f:
                pos = pickle.load(f)
            with open(hg_dir + 'hg.neg', 'rb') as f:
                neg = pickle.load(f)

            hw_dict = dict(zip(hedges, weights))
            pos_hedges_train = pos["train"]
            neg_hedges_train = neg["train"]
            pos_hedges_test = pos["test"]
            neg_hedges_test = neg["test"]

            vis_hedges_train = set(hedges) - set(pos_hedges_train) - set(pos_hedges_test)
            vis_weights_train = [hw_dict[hedge] for hedge in vis_hedges_train]

            vis_hedges_test = set(hedges) - set(pos_hedges_test)
            vis_weights_test = [hw_dict[hedge] for hedge in vis_hedges_test]

            self.pos_hedges_train = list(pos_hedges_train)
            self.neg_hedges_train = list(neg_hedges_train)
            self.vis_hedges_train = list(vis_hedges_train)
            self.vis_weights_train = list(vis_weights_train)

            self.pos_hedges_test = list(pos_hedges_test)
            self.neg_hedges_test = list(neg_hedges_test)
            self.vis_hedges_test = list(vis_hedges_test)
            self.vis_weights_test = list(vis_weights_test)

            self.hedge_size = hedge_size
            self.neg_type = neg_type
            self.imb = imb
            self.seed = seed

            print("..dond")
            return

        print("Generating pos neg hedges..")

        hedges = self.hedges
        weights = self.weights

        random.seed(seed)  # random seed

        hw_dict = dict(zip(hedges, weights))

        n_hedges = len(hedges)

        # Positive hedges
        s_hedges = set()
        if hedge_size is None:  # arb size
            print(n_hedges)
            n_pos = get_n_pos(n_hedges, n_candidates=n_hedges)
            s_hedges = random.sample(hedges, k=n_pos)
            pos_hedges = s_hedges
            n_s_hedges = len(s_hedges)

        else:  # target size
            for idx, hedge in enumerate(hedges):
                if len(hedge) == hedge_size:
                    s_hedges.add(hedge)

            n_s_hedges = len(s_hedges)
            n_pos = get_n_pos(n_hedges, n_s_hedges)
            pos_hedges = random.sample(s_hedges, k=n_pos)

        # Negative hedges
        print("\n..neg..")
        pg_all = hedges_to_pg(hedges=hedges, weights=weights, p=2)
        n_neg = get_n_neg(n_pos, class_imb=imb)

        if hedge_size is None:
            if neg_type == "hub":
                neg_hedges = get_arbsize_hubs(ARB_RANGE, G=pg_all, exclude=s_hedges, n=n_neg, strong=False)
            elif neg_type == "clique":
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            if neg_type == "hub":
                neg_hedges = get_hubs(hedge_size, G=pg_all, exclude=s_hedges, n=n_neg, strong=False)
            elif neg_type == "clique":
                neg_hedges = get_cliques(hedge_size, G=pg_all, exclude=s_hedges, n=n_neg, strong=False)
            else:
                raise NotImplementedError

        n_neg = len(neg_hedges)

        print("Target hedge size: {}".format(hedge_size))
        print("# {}-size hedges: {}/{}".format(hedge_size, n_s_hedges, n_hedges))
        print("# pos: {}".format(n_pos))
        print("# neg: {}".format(n_neg))
        print('pos:neg = {}'.format(n_pos / float(n_neg)))

        # Train test split
        pos_hedges_train = random.sample(pos_hedges, k=int(n_pos * 0.5))
        neg_hedges_train = random.sample(neg_hedges, k=int(n_neg * 0.5))

        pos_hedges_test = set(pos_hedges) - set(pos_hedges_train)
        neg_hedges_test = set(neg_hedges) - set(neg_hedges_train)

        # Visible hedges
        vis_hedges_train = set(hedges) - set(pos_hedges_train) - set(pos_hedges_test)
        vis_hedges_test = set(hedges) - set(pos_hedges_test)

        # Visible weights
        vis_weights_train = [hw_dict[hedge] for hedge in vis_hedges_train]
        vis_weights_test = [hw_dict[hedge] for hedge in vis_hedges_test]

        assert (set(pos_hedges_train).isdisjoint(neg_hedges_train))
        assert (set(pos_hedges_train).isdisjoint(vis_hedges_train))
        assert (set(pos_hedges_test).isdisjoint(neg_hedges_test))
        assert (set(pos_hedges_test).isdisjoint(vis_hedges_test))

        # Test train split
        self.pos_hedges_train = list(pos_hedges_train)
        self.neg_hedges_train = list(neg_hedges_train)
        self.vis_hedges_train = list(vis_hedges_train)
        self.vis_weights_train = list(vis_weights_train)

        self.pos_hedges_test = list(pos_hedges_test)
        self.neg_hedges_test = list(neg_hedges_test)
        self.vis_hedges_test = list(vis_hedges_test)
        self.vis_weights_test = list(vis_weights_test)

        self.hedge_size = hedge_size
        self.neg_type = neg_type
        self.imb = imb
        self.seed = seed

        print("..done")

    def generate_pgs(self):

        print("Generating pgs..")

        hedge_size = self.hedge_size

        vis_hedges_train = self.vis_hedges_train
        vis_weights_train = self.vis_weights_train

        vis_hedges_test = self.vis_hedges_test
        vis_weights_test = self.vis_weights_test

        assert (len(vis_hedges_train) != 0)
        assert (len(vis_weights_train) != 0)
        assert (len(vis_hedges_test) != 0)
        assert (len(vis_weights_test) != 0)

        self.pg2_train = hedges_to_pg(vis_hedges_train, vis_weights_train, p=2)
        self.pg2_test = hedges_to_pg(vis_hedges_test, vis_weights_test, p=2)

        self.pg3_train = hedges_to_pg(vis_hedges_train, vis_weights_train, p=3)
        self.pg3_test = hedges_to_pg(vis_hedges_test, vis_weights_test, p=3)

        if hedge_size > 4 or hedge_size is None:
            self.pg4_train = hedges_to_pg(vis_hedges_train, vis_weights_train, p=4)
            self.pg4_test = hedges_to_pg(vis_hedges_test, vis_weights_test, p=4)

        print("..done")

    def print_summary(self):

        print("------------------------------------")
        print("Summary of HyperGraph")
        print("\n")

        print("Task")
        print("graph name: {}".format(self.gn))
        print("target hedge size: {}".format(self.hedge_size))
        print("\n")

        print("Pos/Neg")
        print("neg: {}{}".format(self.neg_type, self.imb))
        print("seed: {}".format(self.seed))
        print("\n")

        print("All")
        print("# transactions: {}".format(len(self.transactions)))
        print("# hedges: {}".format(len(self.hedges)))
        print("\n")

        print("Train")
        print("# vis hedges: {}".format(len(self.vis_hedges_train)))
        print("# pos hedges: {}".format(len(self.pos_hedges_train)))
        print("# neg hedges: {}".format(len(self.neg_hedges_train)))
        print("pos/neg: {}".format(len(self.pos_hedges_train) / float(len(self.neg_hedges_train))))
        print("\n")

        print("Test")
        print("# vis hedges: {}".format(len(self.vis_hedges_test)))
        print("# pos hedges: {}".format(len(self.pos_hedges_test)))
        print("# neg hedges: {}".format(len(self.neg_hedges_test)))
        print("pos/neg: {}".format(len(self.pos_hedges_test) / float(len(self.neg_hedges_test))))
        print("\n")

    def cache_hg(self):

        print("Caching..")

        gn = self.gn
        hedge_size = self.hedge_size
        neg_type = self.neg_type
        imb = self.imb
        seed = self.seed

        dir = hg_dir.format(gn, hedge_size, neg_type, imb, seed)
        if not os.path.exists(dir):
            os.makedirs(dir)

        path = dir + 'hg'

        extensions = ['.pos', '.neg', '.pgs']

        with open(path + extensions[0], 'wb') as f:
            pos = defaultdict(list)
            pos["train"] = self.pos_hedges_train
            pos["test"] = self.pos_hedges_test
            pickle.dump(pos, f)

        with open(path + extensions[1], 'wb') as f:
            neg = defaultdict(list)
            neg["train"] = self.neg_hedges_train
            neg["test"] = self.neg_hedges_test
            pickle.dump(neg, f)

        with open(path + extensions[2], 'wb') as f:
            pgs = defaultdict(dict)
            pgs["pg2_train"] = self.pg2_train
            pgs["pg2_test"] = self.pg2_test

            pgs["pg3_train"] = self.pg3_train
            pgs["pg3_test"] = self.pg3_test

            pgs["pg4_train"] = self.pg4_train
            pgs["pg4_test"] = self.pg4_test
            pickle.dump(pgs, f)

        # write summary
        original = sys.stdout
        with open(path + '-summary.txt', "w") as log:
            sys.stdout = log
            self.print_summary()
            sys.stdout = original

        print("..done")

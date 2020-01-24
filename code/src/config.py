import argparse

MAX_HEDGE_SIZE = 10

ARB_RANGE = [5, 10]
CLASS_IMBS = [10, 5, 2, 1]
DEFAULT_IMB = CLASS_IMBS[0]
MAX_NEG_GENERATION = 10 * 1000 * 1000
MAX_ITER = 10 * 1000 * 1000

dataset_dir = '../../dataset/'
statistics_dir = '../../statistics/'

hg_dir = '../../hg-MAX%s/{}/size-{}/neg-{}{}/seed-{}/' % MAX_HEDGE_SIZE
results_dir = '../../results-MAX%s/size-{}/neg-{}{}/seed-{}/' % MAX_HEDGE_SIZE
plots_dir = '../../plots-MAX%s/' % MAX_HEDGE_SIZE
tables_dir = '../../tables-MAX%s/' % MAX_HEDGE_SIZE
grids_dir = '../../grids-MAX%s/' % MAX_HEDGE_SIZE

interpretations_dir = '../../interpretations-MAX%s-new/' % MAX_HEDGE_SIZE

gns = [
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

features = [
    "gm",
    "hm",
    "am",
    "cn",
    "jc",
    "aa"]

hss = [4, 5]


def parse_args():
    parser = argparse.ArgumentParser(description="Hedgepred")

    parser.add_argument('--hs', default=5)

    parser.add_argument('--neg_type', default="hub", help="hub/star, clique")

    parser.add_argument('--imb', default=DEFAULT_IMB)

    parser.add_argument('--seed', default=0)

    # parser.add_argument('--f', default="gm", help="gm, hm, am, cn, jc, aa")

    parser.add_argument('--gn', default='email-Enron')

    return parser.parse_args()

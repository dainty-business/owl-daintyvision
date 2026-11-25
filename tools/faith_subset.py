"""The fixed TCGA subset of the capped faithfulness replica: 100 case ids
drawn with np.random.default_rng(20261115) from the sorted case ids of tcga/labels.csv.

usage: python faith_subset.py tcga\\labels.csv [--complement]   (one id per line; --complement for extract_tcga --skip)
"""
import csv
import sys

import numpy as np


def subset(labels, n=100, seed=20261115):
    ids = sorted(r["case_id"] for r in csv.DictReader(open(labels)))
    return set(np.random.default_rng(seed).choice(ids, n, replace=False))


if __name__ == "__main__":
    s = subset(sys.argv[1])
    ids = sorted(r["case_id"] for r in csv.DictReader(open(sys.argv[1])))
    print("\n".join(i for i in ids if (i in s) != ("--complement" in sys.argv)))

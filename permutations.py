# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
#import sys
import os
#from numba import jit


def inverse(permutation):
    res = np.empty_like(permutation)
    res[permutation] = np.arange(len(permutation))
    return res


#@jit
def cycles(permutation):
    N = len(permutation)
    labels = np.empty(N, int)
    labels.fill(-1)
    label = 0
    k = 0
    order = np.empty(N, int)
    for i in range(N):
        if labels[i] != -1:
            continue
        while True:
            order[k] = i
            k = k + 1
            labels[i] = label
            i = permutation[i]
            if labels[i] != -1:
                break
        label += 1

    return order, labels

def cycle(permutation, idx):
    res = []
    x = idx
    for _ in range(50000):
        res.append(x)
        x = permutation[x]
        if x == idx:
            return res
    os._exit(1)
    # raise Exception('cycle failed to terminate PILY')





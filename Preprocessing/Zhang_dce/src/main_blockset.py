from __future__ import print_function, division
import sys, time
import pandas as pd
import numpy as np
from RemoveAlgorithms.Basic import adult, dutch, Domains
from RemoveAlgorithms.Utility import Utility
from RemoveAlgorithms.Detection import Judge
from RemoveAlgorithms.Naive import Naive
from RemoveAlgorithms.MGraph import MGraph
from RemoveAlgorithms.MData import MData
from RemoveAlgorithms.HandlingConditionalDiscrimination import LM, LPS, Removeall
from RemoveAlgorithms.Blackbox import DI
from itertools import permutations


def Batch(data, C, E, Qs):
    avg, std, quanlified, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(data, C, E, Qs, 0.05)
    print('=====Block set: %s ========='%' ,'.join(Qs.columns))
    print('aveage:    %0.3f' % avg)
    print('sigma:     %0.3f' % std)
    print('quanlified/non-empty: %d/%d' % (quanlified, nonemptyset))
    print('large set: %d' % largerset)
    print('small set: %d' % smallerset)
    print('max:       %0.3f' % maxvalue)
    print('min:       %0.3f' % minvalue)


if __name__ == '__main__':

    tau = 0.05
    data, C, E, Qs, Xs, Ys = dutch()
    for a1 in [['edulevel'], ['edulevel', 'age']]:
        for per in range(1, 4):
            for a2 in permutations(['countrybirth', 'citizenship', 'prevresidence'], per):
                blockset = a1 + list(a2)
                Qs = Domains(blockset, data)
                Batch(data, C, E, Qs)

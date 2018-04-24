#!/usr/bin/env python3
from markov import *

import numpy as np
from itertools import islice
with open('1999-05-17.txt', 'r') as repfile:
    repcorpus = repfile.read().replace('\n', '')

repalphabet = ' aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz'
repmarkov = markovmodel.fromscratch(2, len(repalphabet))
train_markov_model(repmarkov,
                   list(islice(map_el_to_int(repcorpus, repalphabet), 0, 50000)),
                   max_iterations=100)

print(repmarkov)

#!/usr/bin/env python3
import numpy as np
from markov import *

with open('brown50000.txt', 'r') as brownfile:
    browncorpus = brownfile.read().replace('\n', '')

randbrown_alphabet = ' abcdefghijklmnopqrstuvwxyz'
randbrown_markov = markovmodel.fromscratch(2, len(randbrown_alphabet))
train_markov_model(randbrown_markov,
                   list(map_el_to_int(browncorpus, randbrown_alphabet)),
                   max_iterations=100)

print(randbrown_markov)

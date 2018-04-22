#!/usr/bin/env python3
import numpy as np
from markov import *

with open('brown50000.txt', 'r') as brownfile:
    browncorpus = brownfile.read().replace('\n', '')

alphabet = ' abcdefghijklmnopqrstuvwxyz'
brownmarkov = markovmodel.fromscratch(2, len(alphabet))
train_markov_model(brownmarkov,
                   list(map_el_to_int(browncorpus, alphabet)),
                   max_iterations=100)

print(brownmarkov)

#!/usr/bin/env python3
from itertools import islice
from markov import *

with open('brown50000.txt', 'r') as brownfile:
    corpus = brownfile.read().replace('\n', '')

alphabet = ' abcdefghijklmnopqrstuvwxyz'

observations = list(islice(
    map_el_to_int(corpus, alphabet),
    0, 50000))

model = train_best_markov_model(
    2, len(alphabet),
    observations,
    20,
    4,
    100)

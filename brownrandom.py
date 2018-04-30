#!/usr/bin/env python3
from itertools import islice
from markov import *

with open('brown50000.txt', 'r') as brownfile:
    corpus = brownfile.read().replace('\n', '')

alphabet = 'abcdefghijklmnopqrstuvwxyz '

observations = list(islice(
    map_el_to_int(corpus, alphabet),
    0, 50000))

model = markovmodel.fromscratch(2, len(alphabet))
train_markov_model(model, observations, 100)

scoretable, groups, ungroupables = markov_alphabetical_analysis(model, alphabet)
print(scoretable)
print(groups)

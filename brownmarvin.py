#!/usr/bin/env python3
from numpy import array
from markov import *

marvin_transition = array([[0.47468, 0.52532],
                           [0.51656, 0.48344]])
marvin_observation = array(
    [[0.03688, 0.03735, 0.03408, 0.03455, 0.03828, 0.03782, 0.03922, 0.03688, 0.03408, 0.03875, 0.04062, 0.03735, 0.03968, 0.03548, 0.03735, 0.04062, 0.03595, 0.03641, 0.03408, 0.04062, 0.03548, 0.03922, 0.04062, 0.03455, 0.03595, 0.03408, 0.03408],
     [0.03397, 0.03909, 0.03537, 0.03537, 0.03909, 0.03583, 0.03630, 0.04048, 0.03537, 0.03816, 0.03909, 0.03490, 0.03723, 0.03537, 0.03909, 0.03397, 0.03397, 0.03816, 0.03676, 0.04048, 0.03443, 0.03537, 0.03955, 0.03816, 0.03723, 0.03769, 0.03955]]
)
marvin_initial = array([[0.51316, 0.48684]])

with open('brown50000.txt', 'r') as brownfile:
    corpus = brownfile.read().replace('\n', '')

alphabet = 'abcdefghijklmnopqrstuvwxyz '

model = markovmodel(marvin_transition, marvin_observation, marvin_initial, rel_tol=1e-3)

train_markov_model(model,
                   list(map_el_to_int(corpus, alphabet)),
                   max_iterations=100)

scoretable, groups, ungroupables = markov_alphabetical_analysis(model, alphabet)
print(scoretable)
print(groups)

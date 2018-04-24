#!/usr/bin/env python3
from markov import markovmodel,train_markov_model, map_el_to_int, markov_alphabetical_analysis

with open('brown50000.txt', 'r') as brownfile:
    corpus = brownfile.read().replace('\n', '')

alphabet = ' abcdefghijklmnopqrstuvwxyz'

model = markovmodel.fromscratch(2, len(alphabet))
print(model)
train_markov_model(model,
                   list(map_el_to_int(corpus, alphabet)),
                   max_iterations=100)
table, groups, _ = markov_alphabetical_analysis(model, alphabet)
print(table)
print(groups)

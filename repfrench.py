#!/usr/bin/env python3
from markov import *

from itertools import islice
from markov import *

with open('1999-05-17.txt', 'r') as repfile:
    corpus = repfile.read().replace('\n', '')

alphabet = ' aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz'

observations = list(islice(
    map_el_to_int(corpus, alphabet),
    0, 50000))

model = train_best_markov_model(
    2, len(alphabet),
    observations,
    20,
    4,
    100)

_, scale_factors = alpha_pass(model, observations)
print('score', log_observation_sequence_probability(scale_factors))
def latexify(char):
    if char == ' ':
        return '\\textvisiblespace'
    return char


scoretable, groups, ungroupables = markov_alphabetical_analysis(model, alphabet)
scoretable = [[latexify(line[0]),
               *('${:.3f}$'.format(probas * 100) for probas in line[1:])]
              for line in scoretable]
scoretable.insert(0, ['caractère', 'État 1 (%)', 'État 2 (%)'])
print('#+ATTR_LATEX: :align l l l')
print(orgmodetable(scoretable, header=True), '\n\n\n')

groupstable = [['{ ' + ',  '.join((latexify(char) for char in group)) + ' }'
                  for group in groups] ]
groupstable.insert(0, ['Groupe 1', 'Groupe 2'])

if len(ungroupables) > 0:
    groupstable[0].insert(
        len(ungroupables), 'Hors groupes')
    groupstable[1].insert(
        len(ungroupables), '{ ' + ', '.join(latexify(char) for char in ungroupables) + ' }')
print(orgmodetable(groupstable, header=True))

#!/usr/bin/env python3
from itertools import islice
from markov import *

with open('1999-05-17.txt', 'r') as repfile:
    corpus = repfile.read().replace('\n', '')

alphabet = ' abcdefghijklmnopqrstuvwxyz'

def translate(iterable, translation_table):
    for el in iterable:
        if el in translation_table:
            for tr in translation_table[el]:
                yield tr
        else:
            yield el

translations = {'à': 'a',
                'â': 'a',
                'æ': 'ae',
                'ç': 'c',
                'é': 'e',
                'è': 'e',
                'ê': 'e',
                'ë': 'e',
                'î': 'i',
                'ï': 'i',
                'ô': 'o',
                'œ': 'oe',
                'ù': 'u',
                'û': 'u',
                'ü': 'u',
                'ÿ': 'y',
                '\'': ' ',
                '-': ' '}

observations = list(islice(
    map_el_to_int(translate(corpus, translations), alphabet),
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
print('#+ATTR_LATEX: :align l l l\n',
      '#+CAPTION: répartition des caractères', sep='')
try:
    print('#+NAME:', name + 'rep')
except NameError:
    pass
print(orgmodetable(scoretable, header=True), '\n\n\n')

groupstable = [['{ ' + ',  '.join((latexify(char) for char in group)) + ' }'
                  for group in groups] ]
groupstable.insert(0, ['Groupe 1', 'Groupe 2'])

if len(ungroupables) > 0:
    groupstable[0].insert(
        len(ungroupables), 'Hors groupes')
    groupstable[1].insert(
        len(ungroupables), '{ ' + ', '.join(latexify(char) for char in ungroupables) + ' }')

print('#+CAPTION: groupes formés')
try:
    print('#+NAME:', name + 'grp')
except NameError:
    pass
print(orgmodetable(groupstable, header=True))

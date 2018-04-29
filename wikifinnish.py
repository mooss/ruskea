#!/usr/bin/env python3
from itertools import islice
from markov import *

with open('wiki_fi_50000.txt', 'r') as suofile:
    corpus = suofile.read().replace('\n', '')

alphabet = ' aäåbcdefghijklmnoöpqrstuvwxyz'

observations = list(islice(
    map_el_to_int(corpus, alphabet),
    0, 50000))

model = train_best_markov_model(
    2, len(alphabet),
    observations,
    nb_candidates=3,
    train_iter=8,
    max_iter=100)

name = 'finnish'
descr = ' - Wikipédia finnois'
_, scale_factors = alpha_pass(model, observations)
print('score', log_observation_sequence_probability(scale_factors))

def orgmodetable(matrix, header=False):
    maxlen = [0] * len(matrix[0])
    for line in matrix:
        for i, cell in enumerate(line):
            if len(maxlen) <= i or len(str(cell)) > maxlen[i]:
                maxlen[i] = len(str(cell))

    def orgmodeline(line, fill=' '):
        joinsep = fill + '|' + fill
        return '|' + fill + joinsep.join(
            str(cell) + fill * (mlen - len(str(cell)))
            for cell, mlen in zip(line, maxlen)
        ) + fill + '|'

    result = ''
    if header:
        result = orgmodeline(matrix[0]) + '\n' + \
            orgmodeline(('-') * len(maxlen), fill='-') + '\n'
        matrix = matrix[1:]
    result += '\n'.join(orgmodeline(line) for line in matrix)
    return result


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
caption = '#+CAPTION: Répartition des caractères'

try:
    descr
    caption = caption + descr
except NameError:
    pass
print(caption)

try:
    name
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

caption = '#+CAPTION: Groupes formés'
try:
    descr
    caption = caption + descr
except NameError:
    pass
print(caption)

try:
    name
    print('#+NAME:', name + 'grp')
except NameError:
    pass
print(orgmodetable(groupstable, header=True))

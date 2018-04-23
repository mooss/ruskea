#!/usr/bin/env python3
from markov import *

markovtest = markovmodel.fromscratch(3, 4)
print(markovtest.transition_matrix)

try:
    markovtemperature = markovmodel(
        np.matrix([[0.7, 0.3],
                   [0.4, 0.6]]),
        np.matrix([[0.1, 0.4, 0.5],
                   [0.7, 0.2, 0.1]]),
        np.matrix([[0.6, 0.4]])
    )
    print('transition:', markovtemperature.transition_matrix,
          'observation:', markovtemperature.observation_matrix,
          'initial states:', markovtemperature.initial_state_distribution,
          sep='\n')
except Exception as e:
    print('construction failed:', str(e))

observations = [0, 1, 0, 2]
alpha_matrix, scales = alpha_pass(markovtemperature, observations)
print(alpha_matrix)
print(scales)

beta_matrix = beta_pass(markovtemperature, observations, scales)
print(beta_matrix)

gamma, digamma = gamma_digamma_pass(
    markovtemperature,
    observations,
    alpha_matrix,
    beta_matrix
)
print(gamma, '\n\n\n', digamma, sep='')

gamma2, digamma2, scale_factors = greek_pass(markovtemperature, observations)
if not np.array_equal(gamma, gamma2) or not np.array_equal(digamma, digamma2):
    print('gammas or digammas from greek_pass and from gamma_digamma_pass differ')
else:
    print('gammas and digammas from greek_pass and from gamma_digamma_pass are the same')

if not np.array_equal(scales, scale_factors):
    print('the scale factors from alpha_pass et greek_pass differ')
else:
    print('the scale factors from alpha_pass et greek_pass are the same')

from copy import deepcopy
markov_copy = deepcopy(markovtemperature)
print(markov_copy)
train_markov_model(markov_copy, observations, 10)
print(markov_copy)

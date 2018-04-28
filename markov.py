import math
import random
from numpy import zeros, full, array
from copy import deepcopy

def stochastic_variation(mat, epsilon):
    """Slightly changes the values of a matrix while making sure that the sum of the rows are kept the same.

    Parameters
    ----------
    mat : np.matrix
        Matrix to change.

    epsilon : float
        Maximal variation.
    """
    random.seed()
    for row in mat:
        delta = 0
        for i in range(0, len(row)):
            # if delta > epsilon / 2:
            #     nextvariation = random.uniform(-epsilon, 0)
            # elif delta < -epsilon / 2:
            #     nextvariation = random.uniform(0, epsilon)
            # else:
            #     nextvariation = random.uniform(-epsilon, epsilon)
            if random.uniform(0, 1) >= .5:
                row[i] += random.uniform(*epsilon) #nextvariation
            else:
                row[i] -= random.uniform(*epsilon)

            if row[i] < 0:
                row[i] = -row[i]
                # delta += nextvariation

        factor = 1/sum(row)
        for i in range(0, len(row)):
            row[i] *= factor
        #     nextvalue = random.gauss(row[i], epsilon)
        #     delta += nextvalue - row[i]
        #     row[i] = nextvalue
        # meandelta = delta/len(row)
        # for i in range(0, len(row)):
        #     row[i] -= meandelta


def prob_matrix(M, p_range):
    try:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if random.uniform(0, 1) >= .5:
                    M[i][j] += random.uniform(p_range[0], p_range[1])
                else:
                    M[i][j] -= random.uniform(p_range[0], p_range[1])
        for i in range(M.shape[0]):
            factor = M[i].sum()
            for j in range(M.shape[1]):
                M[i][j] *= 1/factor
    except:
        for j in range(M.shape[0]):
            if random.uniform(0, 1) >= .5:
                M[j] += random.uniform(p_range[0], p_range[1])
            else:
                M[j] -= random.uniform(p_range[0], p_range[1])
        factor = M.sum()
        for j in range(M.shape[0]):
            M[j] *= 1/factor
    return M


class markovmodel(object):
    def fromscratch(N, M):
        """Create a Markov model from scratch with the following matrices dimensions:
         - A is NxN
         - B is NxM
         - PI is 1xN

        Parameters
        ----------
        N : int

        M : int

        Returns
        -------
        out : The corresponding Markov model
        """
        inverseN = 1 / N
        inverseM = 1 / M

        transition = full((N, N), inverseN)
        observation = full((N, M), inverseM)
        initial = full((1, N), inverseN)

        # prob_matrix(transition, (0.001, 0.005))
        # prob_matrix(observation, (0.02, 0.025))
        stochastic_variation(transition, (0.000, 0.005))
        stochastic_variation(observation, (0.02, 0.025))
        stochastic_variation(initial, (0.001, 0.005))

        return markovmodel(transition, observation, initial)

    def __init__(self,
                 transition_matrix,
                 observation_matrix,
                 initial_state_distribution,
                 rel_tol=1e-9):
        """Create a markov model.

        Parameters
        ----------
        transition_matrix : np.matrix
            NxN matrix containing the state transitions probabilities.

        observation_matrix : np.matrix
            NxM matrix containing the observation probabilities.

        initial_state_distribution : np.matrix
            1xN matrix containing the initial state distribution
        """
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.initial_state_distribution = initial_state_distribution
        self.rel_tol = rel_tol
        self.ensure_dimensional_validity()
        self.ensure_row_stochasticity()

        self.ndim = transition_matrix.shape[0]
        self.mdim = observation_matrix.shape[1]

    def __str__(self):
        return '\n'.join((
            'transition:',
            str(self.transition_matrix), '',
            'observation:',
            str(self.observation_matrix), '',
            'initial states:',
            str(self.initial_state_distribution)))

    def ensure_dimensional_validity(self):
        """Raises an exception if the matrices' dimensions are not right.
        """
        tr_rows, tr_columns = self.transition_matrix.shape
        ob_rows, _ = self.observation_matrix.shape
        in_rows, in_columns = self.initial_state_distribution.shape

        if not (tr_rows == tr_columns == ob_rows == in_columns):
            raise ValueError('The number of transition rows, transition columns, observation rows and initial state distribution columns is not the same')

        if in_rows != 1:
            raise ValueError("The initial state distribution matrix should have one and only one row")

    def ensure_row_stochasticity(self):
        """Raises an exception if the matrices are not row-stochastic.
        """
        def fullofones(iterable):
            return all(math.isclose(el, 1, rel_tol = self.rel_tol) for el in iterable)

        if not fullofones(self.transition_matrix.sum(axis=1)):
            raise ValueError("The transition matrix is not row stochastic")

        if not fullofones(self.observation_matrix.sum(axis=1)):
            raise ValueError("The observation matrix is not row stochastic")

        if not fullofones(self.initial_state_distribution.sum(axis=1)):
            raise ValueError("The initial_state_distribution matrix is not row stochastic")

    def getinitialstate(self, i):
        return self.initial_state_distribution[0,i]

def alpha_pass(markov, observations):
    """Implementation of the forward algorithm to compute the alpha_t values.

    Parameters
    ----------
    markov : markovchain

    observations : iterable

    Returns
    -------
    out : np.array
        The alpha_t values.
    """
    alpha = zeros(shape=(len(observations), markov.ndim))
    scale_factors = zeros(shape=(len(observations)))

    # alpha_zero initialization

    for i in range(0, markov.ndim):
        alpha[0, i] = markov.getinitialstate(i) * markov.observation_matrix[i, 0]
        scale_factors[0] += alpha[0, i]

    scale_factors[0] = 1 /scale_factors[0]

    for i in range(0, markov.ndim):
        alpha[0, i] *= scale_factors[0]

    # alpha_t computation
    for t in range(1, len(observations)):
        for i in range(0, markov.ndim):
            for j in range(0, markov.ndim):
                alpha[t, i] += alpha[t - 1, j] * markov.transition_matrix[j, i]
            alpha[t, i] *= markov.observation_matrix[i, observations[t]]
            scale_factors[t] += alpha[t, i]

        # scale alpha
        scale_factors[t] = 1 / scale_factors[t]
        for i in range(0, markov.ndim):
            alpha[t, i] *= scale_factors[t]

    return (alpha, scale_factors)

def beta_pass(markov, observations, scale_factors):
    """

    Parameters
    ----------
    markov : 

    observations : 

    Returns
    -------
    out : 

    """
    beta = zeros(shape=(len(observations), markov.ndim))

    # all elements of the last column take the last scale factor as value
    # np.vectorize(lambda _: scale_factors[-1])(beta.transpose()[-1])
    # for line in beta:
    #     line[-1] = scale_factors[-1]
    for i in range(0, markov.ndim):
        beta[-1, i] = scale_factors[-1]

    for t in reversed(range(0, len(observations) - 1)):
        for i in range(0, markov.ndim):
            for j in range(0, markov.ndim):
                beta[t, i] += markov.transition_matrix[i, j] * markov.observation_matrix[j, observations[t+1]] * beta[t + 1, j]

            # scale beta
            beta[t, i] *= scale_factors[t]

    return beta

def gamma_digamma_pass(markov, observations, alpha, beta):
    """

    Parameters
    ----------
    markov : 

    observations : 

    alpha : 

    beta : 

    Returns
    -------
    out : 

    """
    digamma = zeros(shape=(len(observations), markov.ndim, markov.ndim))
    gamma = zeros(shape=(len(observations), markov.ndim))

    for t in range(0, len(observations) - 1):
        for i in range(0, markov.ndim):
            for j in range(0, markov.ndim):
                digamma[t, i, j] = alpha[t, i] * markov.transition_matrix[i, j] * markov.observation_matrix[j, observations[t + 1]] * beta[t + 1, j]
                gamma[t, i] += digamma[t, i, j]

    # special case for the last gammas
    for i in range(0, markov.ndim - 1):
        gamma[-1, i] = alpha[-1, i]

    return (gamma, digamma)

def greek_pass(markov, observations):
    """

    Parameters
    ----------
    markov : 

    observations : 

    Returns
    -------
    out : 

    """
    alpha, scale_factors = alpha_pass(markov, observations)
    beta = beta_pass(markov, observations, scale_factors)
    return (*gamma_digamma_pass(markov, observations, alpha, beta), scale_factors)

def reestimate_initial_state_distribution(markov, gamma):
    """Use previously-calculated gamma values to do a re-estimation of the initial state distribution.

    Parameters
    ----------
    markov : 

    gamma : 

    Returns
    -------
    out : 
    """
    for i in range(0, markov.ndim):
        markov.initial_state_distribution[0, i] = gamma[0, i]

def reestimate_transition_matrix(markov, gamma, digamma):
    """


        Parameters
        ----------
        markov : 

        gamma : 

        digamma : 

        Returns
        -------
        out : 

    """
    for i in range(0, markov.ndim):
        for j in range(0, markov.ndim):
            gamma_acc, digamma_acc = 0, 0
            for t in range(0, len(gamma) - 1):
                gamma_acc += gamma[t, i]
                digamma_acc += digamma[t, i, j]
            markov.transition_matrix[i, j] = digamma_acc / gamma_acc

    markov.ensure_row_stochasticity()

def reestimate_observation_matrix(markov, observations, gamma):
    """

    Parameters
    ----------
    markov : 

    observations : 

    gamma : 
    """
    for i in range(0, markov.ndim):
        for j in range(0, markov.mdim):
            gamma_acc_observed, gamma_acc_all = 0, 0
            for t in range(0, len(observations)):
                if observations[t] == j:
                    gamma_acc_observed += gamma[t, i]
                gamma_acc_all += gamma[t, i]
            markov.observation_matrix[i, j] = gamma_acc_observed / gamma_acc_all

def log_observation_sequence_probability(scale_factors):
    """Compute the log of the observation's sequence probability according to a markov model, using the scales factors.

    Parameters
    ----------
    scale_factors : 

    Returns
    -------
    out : 
    """
    result = 0
    for i in range(0, len(scale_factors)):
        result += math.log(scale_factors[i])
    return -result

def reestimate_markov_model(markov, observations):
    """

    Parameters
    ----------
    markov : 

    observations : 

    Returns
    -------
    out : 
    """
    gamma, digamma, scale_factors = greek_pass(markov, observations)
    reestimate_initial_state_distribution(markov, gamma)
    reestimate_transition_matrix(markov, gamma, digamma)
    reestimate_observation_matrix(markov, observations, gamma)
    return log_observation_sequence_probability(scale_factors)

def train_markov_model(markov, observations, max_iterations=200):
    """

    Parameters
    ----------
    markov : 

    observations : 

    max_iterations : 

    Returns
    -------
    out : 
    """
    _, scale_factors = alpha_pass(markov, observations)
    bestlogprob = log_observation_sequence_probability(scale_factors)
    bestmodel = deepcopy(markov)

    for i in range(1, max_iterations):
        logprob = reestimate_markov_model(markov, observations)
        markov.ensure_row_stochasticity()
        if logprob > bestlogprob:
            bestmodel = deepcopy(markov)
            bestlogprob = logprob

    markov = deepcopy(bestmodel)
    return bestlogprob

def train_best_markov_model(N, M, observations, nb_candidates, train_iter, max_iter):
    bestmodel = markovmodel.fromscratch(N, M)
    bestprob = train_markov_model(bestmodel, observations, train_iter)

    for i in range(0, nb_candidates - 1):
        candidate = markovmodel.fromscratch(N, M)
        candidateprob = train_markov_model(candidate, observations, train_iter)

        if candidateprob > bestprob:
            bestprob = candidateprob
            bestmodel = deepcopy(candidate)

    print(bestprob)
    print(bestmodel)
    train_markov_model(bestmodel, observations, max_iter - train_iter)
    return bestmodel

def map_el_to_int(iterable, alphabet):
    """Map all the elements of an iterable to their index in an alphabet.
    If an element is not in the alphabet, it will be ignored.

    Parameters
    ----------
    iterable : iterable
        The iterable to map.

    alphabet : str
        The letters to keep.

    Returns
    -------
    out : list of int
        The list containing the index of each character in the input string.
    """
    indexation = {letter: index for index, letter in enumerate(alphabet)}
    return (indexation[char] for char in iterable if char in alphabet)

def markov_alphabetical_analysis(markov, alphabet):
    observation_scores = [[letter,
                           *(markov.observation_matrix[state, index]
                              for state in range(0, markov.ndim))]
                          for index, letter in enumerate(alphabet)]

    letter_groups = [list() for _ in range(0, markov.ndim)]
    ungroupables = []

    for letterindex, letter in enumerate(alphabet):
        maxindex = 0
        for state in range(1, markov.ndim):
            if markov.observation_matrix[state, letterindex] >\
               markov.observation_matrix[maxindex, letterindex]:
                maxindex = state
            if markov.observation_matrix[maxindex, letterindex] == 0:
                ungroupables.append(letter)
            else:
                letter_groups[maxindex].append(letter)

    return observation_scores, letter_groups, ungroupables

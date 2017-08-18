"""Hidden Markov Model Toolkit
with the fundamental tasks --
randomly generating data,
finding the best state sequence for observation,
computing the probability of observations through the forward and backward algorithms,
supervised training of HMMs from fully observed data,
and unsupervised training with the Baum-Welch EM algorithm
from data where state sequences are not observed.
"""

import numpy
import numpy.random
import random
import argparse
import codecs
import os
from collections import defaultdict

# define helper functions here
def normalize(countdict):
    """given a dictionary mapping items to counts,
    return a dictionary mapping items to their normalized (relative) counts
    Example: normalize({'a': 2, 'b': 1, 'c': 1}) -> {'a': 0.5, 'b': 0.25, 'c': 0.25}
    """
    total = float(sum(countdict.values()))
    return {item: val/total for item, val in countdict.items()}

def read_arcs(filename):
    """Load parameters from file and store in nested dictionary.
    Assume probabilities are already normalized.
    Return dictionary and boolean indicating whether probabilities were provided.
    """
    arcs = defaultdict(lambda : defaultdict(float))  # will assign 0 to missing keys automatically
    provided = True
    for line in map(lambda line: line.split(), codecs.open(filename, 'r', 'utf8')):
        if len(line)<2:
            continue
        from_s = line[0]
        to_s = line[1]
        if len(line)==3:
            prob = float(line[2])
        else:
            prob = None
            provided = False
        arcs[from_s][to_s] = prob

    return arcs, provided

def write_arcs(arcdict, filename):
    """write dictionary of conditional probabilities to file
    """
    o = codecs.open(filename, 'w', 'utf8')
    for from_s in arcdict:
        for to_s in arcdict[from_s]:
            if arcdict[from_s][to_s]!=0:
                o.write(from_s+' '+to_s+' '+str(arcdict[from_s][to_s])+'\n')
    o.close()

def sample_from_dist(d):
    """given a dictionary representing a discrete probability distribution
    (keys are atomic outcomes, values are probabilities)
    sample a key according to the distribution.
    Example: if d is {'H': 0.7, 'T': 0.3}, 'H' should be returned about 0.7 of time.
    """
    roll = numpy.random.random()
    cumul = 0
    for k in d:
        cumul += d[k]
        if roll < cumul:
            return k

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

def load_observations(filename):
    lines = [line.split() for line in codecs.open(filename, 'r', 'utf8').readlines()]
    if len(lines)%2==1:  # remove extra lines
        lines[:len(lines)-1]
    return [Observation(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]

# hmm model
class HMM:
    def __init__(self, transitions=None, emissions=None):
        """creates a model from transition and emission probabilities"""
        self.transitions = transitions
        self.emissions = emissions
        if self.emissions:
            self.states = self.emissions.keys()

    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities if given.
        Initializes probabilities randomly if unspecified."""
        # TODO: fill in for section a
        self.transitions, tprovided = read_arcs(basename+'.trans')
        self.emissions, eprovided = read_arcs(basename+'.emit')
        self.states = self.emissions.keys()

        # initialize with random parameters if probs were not specified
        if not tprovided:   # at least one transition probability not given in file
            print 'Transition probabilities not given: initializing randomly.'
            self.init_transitions_random()
        if not eprovided:   # at least one emission probability not given in file
            print 'Emission probabilities not given: initializing randomly.'
            self.init_emissions_random()

    def init_transitions_random(self):
        """assign random probability values to the HMM transition parameters
        """
        # Do not modify this function
        for from_state in self.transitions:
            random_probs = numpy.random.random(len(self.transitions[from_state]))
            total = sum(random_probs)
            for to_index, to_state in enumerate(self.transitions[from_state]):
                self.transitions[from_state][to_state] = random_probs[to_index]/total

    def init_emissions_random(self):
        for state in self.emissions:
            random_probs = numpy.random.random(len(self.emissions[state]))
            total = sum(random_probs)
            for symi, sym in enumerate(self.emissions[state]):
                self.emissions[state][sym] = random_probs[symi]/total

    def dump(self, basename):
        """store HMM model parameters in basename.trans and basename.emit"""
        # TODO: fill in for section a
        write_arcs(self.transitions, basename+'.trans')
        write_arcs(self.emissions, basename+'.emit')

    def generate(self, n):
        """return a list of n symbols by randomly sampling from this HMM.
        """
        # TODO: fill in for section c
        states = []
        outputs = []
        for i in range(n):
            if i==0:
                state = sample_from_dist(self.transitions['#'])
            else:
                state = sample_from_dist(self.transitions[state])  # 'state' was generated last time
            states.append(state)
            symbol = sample_from_dist(self.emissions[state])
            outputs.append(symbol)
        return Observation(states, outputs)

    def viterbi(self, observation):
        """given an observation,
        set its state sequence to be the most likely state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        # TODO: fill in for section d
        viterbi_path = []

        viterbi_costs = numpy.zeros((len(self.states), len(observation)))
        viterbi_backpointers = numpy.zeros((len(self.states), len(observation)), dtype=int)

        for oi, obs in enumerate(observation.outputseq):
            for si, state in enumerate(self.states):
                if oi==0:
                    viterbi_costs[si, oi] = self.transitions['#'][state] * self.emissions[state][obs]
                else:
                    best_costs = {}
                    for pi, prevstate in enumerate(self.states):
                        best_costs[pi] = viterbi_costs[pi, oi-1] * self.transitions[prevstate][state]

                    best_state, best_cost = max(best_costs.items(), key=lambda (state, cost): cost)
                    viterbi_costs[si, oi] =  best_cost * self.emissions[state][obs]
                    viterbi_backpointers[si, oi] = best_state

        oi = len(observation)-1
        best_state = numpy.argmax(viterbi_costs[:, oi])
        viterbi_path.append(self.states[best_state])
        while oi>0:
            best_state = viterbi_backpointers[best_state, oi]
            viterbi_path.append(self.states[best_state])
            oi-=1

        observation.stateseq = viterbi_path[::-1]

    def forward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the forward algorithm parameters alpha_i(t)
        for all 0<=t<T and i HMM states.
        """
        # TODO: fill in for section e
        forward_matrix = numpy.zeros((len(self.states), len(observation)))

        for oi, obs in enumerate(observation.outputseq):
            for si, state in enumerate(self.states):
                if oi==0:
                    forward_matrix[si, oi] = self.transitions['#'][state] * self.emissions[state][obs]
                else:
                    for pi, prevstate in enumerate(self.states):
                        forward_matrix[si, oi] += forward_matrix[pi, oi-1] * self.transitions[prevstate][state]

                    forward_matrix[si, oi] *= self.emissions[state][obs]  # factor out common emission prob

        return forward_matrix

    def forward_probability(self, observation):
        """return probability of observation, computed with forward algorithm.
        """
        # TODO: fill in for section d
        forward_matrix = self.forward(observation)
        # sum of forward probabilities in last time step, over all states
        return sum(forward_matrix[:, len(observation)-1])

    def backward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the backward algorithm parameters beta_i(t)
        for all 0<=t<T and i HMM states.
        """
        # TODO: fill in for section e
        backward_matrix = numpy.zeros((len(self.states), len(observation)))
        for si, state in enumerate(self.states):
            backward_matrix[si, len(observation)-1] = 1  # 1s in last column

        for oi in range(len(observation)-2, -1, -1):
            for si, state in enumerate(self.states):
                for ni, nextstate in enumerate(self.states):
                    backward_matrix[si, oi] += backward_matrix[ni, oi+1] * self.transitions[state][nextstate] * self.emissions[nextstate][observation.outputseq[oi+1]]

        return backward_matrix

    def backward_probability(self, observation):
        """return probability of observation, computed with backward algorithm.
        """
        # TODO: fill in for section e
        backward_matrix = self.backward(observation)
        backprob = 0.0  # total probability
        for si, state in enumerate(self.states):
            # prob of transitioning from # to state and giving out observation[0]
            backprob += self.transitions['#'][state] * self.emissions[state][observation.outputseq[0]] * backward_matrix[si, 0]
        return backprob

    def learn_supervised(self, corpus, emitlock=False, translock=False):
        """Given a corpus, which is a list of observations
        with known state sequences,
        set the HMM parameters that maximize the corpus likelihood.
        Do not update the transitions if translock is True,
        or the emissions if emitlock is True.
        """
        # TODO: fill in for section b
        transcounts = defaultdict(lambda : defaultdict(int))
        emitcounts = defaultdict(lambda : defaultdict(int))
        for observation in corpus:
            for oi in range(len(observation)):
                if oi==0:
                    transcounts['#'][observation.stateseq[oi]] += 1
                else:
                    transcounts[observation.stateseq[oi-1]][observation.stateseq[oi]] += 1
                emitcounts[observation.stateseq[oi]][observation.outputseq[oi]] += 1

        self.maximization(emitcounts, transcounts, emitlock, translock)

    def maximization(self, emitcounts, transcounts, emitlock, translock):
        """M-Step: set self.emissions and self.transitions
        conditional probability parameters to be the normalized
        counts from emitcounts and transcounts respectively.
        Do not update if self.emissions if the emitlock flag is True,
        or self.transitions if translock is True.
        """
        if not translock:
            for from_state in transcounts:
                self.transitions[from_state] = normalize(transcounts[from_state])

        if not emitlock:
            for state in emitcounts:
                self.emissions[state] = normalize(emitcounts[state])

    def expectation(self, corpus):
        """E-Step: given a corpus, which is list of observations,
        calculate the expected number of each transition and emission,
        as well as the log likelihood of the observations under the current parameters.
        return a list with the log likelihood, expected emission counts, and expected transition counts.
        """
        log_likelihood = 0.0  # holds running total of the log likelihood of all observations
        emitcounts = defaultdict(lambda : defaultdict(float))  # expected emission counts
        transcounts = defaultdict(lambda : defaultdict(float)) # expected transition counts

        for observation in corpus:
            forward_matrix = self.forward(observation)
            backward_matrix = self.backward(observation)
            forward_prob = sum(forward_matrix[:, len(observation)-1])
            for oi, obs in enumerate(observation.outputseq):
                #emission soft counts
                prob_state = {}
                for si, state in enumerate(self.states):
                    emitcounts[state][obs] += forward_matrix[si, oi] * backward_matrix[si, oi] / forward_prob
                #transition soft counts
                prob_state1_state2 = {}
                for si, state in enumerate(self.states):
                    if oi==0:
                        transcounts['#'][state] += forward_matrix[si, oi] * backward_matrix[si, oi] / forward_prob
                    else:
                        for pi, prevstate in enumerate(self.states):
                            transcounts[prevstate][state] += forward_matrix[pi, oi-1] * self.transitions[prevstate][state] * self.emissions[state][obs] * backward_matrix[si, oi] / forward_prob

            log_likelihood += numpy.log2(forward_prob)

        return log_likelihood, emitcounts, transcounts

    def learn_unsupervised(self, corpus, convergence=0.001, emitlock=False, translock=False, restarts=0):
        """Given a corpus,
        which is a list of observations with the state sequences unknown,
        apply the Baum Welch EM algorithm
        to learn the HMM parameters that maximize the corpus likelihood.
        Do not update the transitions if translock is True,
        or the emissions if emitlock is True.
        Stop when the log likelihood changes less than the convergence threhshold,
        and return the final log likelihood.
        If restarts>0, re-run EM with random initializations.
        """
        # TODO: fill in for section f
        best_log_likelihood = -numpy.inf
        best_model_trans = None
        best_model_emit = None

        for i in range(restarts+1):
            if i>0:
                print "RESTART", i+1, 'of', restarts+1

                if not translock:
                    self.init_transitions_random()
                    print 'Re-initializing transition probabilities'
                if not emitlock:
                    self.init_emissions_random()
                    print 'Re-initializing emission probabilities'

            old_ll = -numpy.inf
            log_likelihood = -1e210  # almost -inf

            while log_likelihood-old_ll > convergence:
                old_ll = log_likelihood
                log_likelihood, emitcounts, transcounts = self.expectation(corpus) # E Step
                self.maximization(emitcounts, transcounts, emitlock, translock)  # M Step
                print 'LOG LIKELIHOOD:', log_likelihood,
                print 'DIFFERENCE:', log_likelihood-old_ll
            print 'CONVERGED'

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_model_trans = self.transitions
                best_model_emit = self.emissions

        self.__init__(best_model_trans, best_model_emit)
        return best_log_likelihood

def main():
    # Do not modify this function
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('paramfile', type=str, help='basename of the HMM parameter file')
    parser.add_argument('function',
                        type=str,
                        choices = ['g', 'v', 'f', 'b', 'sup', 'unsup'],
                        help='random generation (g), best state sequence (v), forward probability of observations (f), backward probability of observations (b), supervised learning (sup), or unsupervised learning (unsup)?')
    parser.add_argument('obsfile', type=str, help='file with list of observations')

    # optional arguments
    parser.add_argument('--convergence', type=float, default=0.1, help='convergence threshold for EM')
    parser.add_argument('--restarts', type=int, default=0, help='number of random restarts for EM')
    parser.add_argument('--emitlock', type=bool, default=False, help='should the emission parameters be frozen during EM training?')
    parser.add_argument('--translock', type=bool, default=False, help='should the transition parameters be frozen during EM training?')

    args = parser.parse_args()

    # initialize model and read data
    model = HMM()
    model.load(args.paramfile)

    if args.function == 'v':
        corpus = load_observations(args.obsfile)
        outputfile = os.path.splitext(args.obsfile)[0]+'.tagged.obs'

        with codecs.open(outputfile, 'w', 'utf8') as o:
            for observation in corpus:
                model.viterbi(observation)
                o.write(str(observation))

    elif args.function == 'f':
        corpus = load_observations(args.obsfile)
        outputfile = os.path.splitext(args.obsfile)[0]+'.forwardprob'
        with open(outputfile, 'w') as o:
            for observation in corpus:
                o.write(str(model.forward_probability(observation))+'\n')

    elif args.function == 'b':
        corpus = load_observations(args.obsfile)
        outputfile = os.path.splitext(args.obsfile)[0]+'.backwardprob'
        with open(outputfile, 'w') as o:
            for observation in corpus:
                o.write(str(model.backward_probability(observation))+'\n')

    elif args.function == 'sup':
        corpus = load_observations(args.obsfile)
        model.learn_supervised(corpus, args.emitlock, args.translock)
        #write the trained model
        corpusbase = os.path.splitext(os.path.basename(args.obsfile))[0]
        model.dump(args.paramfile+'.'+corpusbase+'.trained')

    elif args.function == 'unsup':
        corpus = load_observations(args.obsfile)
        log_likelihood = model.learn_unsupervised(corpus, args.convergence, args.emitlock, args.translock, args.restarts)
        #write the trained model
        print "The final model's log likelihood is", log_likelihood
        corpusbase = os.path.splitext(os.path.basename(args.obsfile))[0]
        model.dump(args.paramfile+'.'+corpusbase+'.trained')

    elif args.function == 'g':
        # write randomly generated sentences
        with codecs.open(args.obsfile, 'w', 'utf8') as o:
            for _ in range(20):
                o.write(str(model.generate(random.randint(1, 15)))) # random-length sentences

if __name__=='__main__':
    main()

"""Hidden Markov Model Toolkit
with the fundamental tasks --
randomly generating data,
finding the best state sequence for observation,
computing the probability of observations through the forward and backward algorithms,
supervised training of HMMs from fully observed data,
and unsupervised training with the Baum-Welch EM algorithm
from data where state sequences are not observed.
"""

import random
import argparse
import codecs
import os

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
        pass

    def dump(self, basename):
        """store HMM model parameters in basename.trans and basename.emit"""
        # TODO: fill in for section a
        pass

    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM.
        """
        # TODO: fill in for section c
        pass

    def viterbi(self, observation):
        """given an observation,
        set its state sequence to be the most likely state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        # TODO: fill in for section d
        pass

    def forward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the forward algorithm parameters alpha_i(t)
        for all 0<=t<T and i HMM states.
        """
        # TODO: fill in for section e
        pass

    def forward_probability(self, observation):
        """return probability of observation, computed with forward algorithm.
        """
        # TODO: fill in for section e
        pass

    def backward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the backward algorithm parameters beta_i(t)
        for all 0<=t<T and i HMM states.
        """
        # TODO: fill in for section e
        pass

    def backward_probability(self, observation):
        """return probability of observation, computed with backward algorithm.
        """
        # TODO: fill in for section e
        pass

    def learn_supervised(self, corpus, emitlock=False, translock=False):
        """Given a corpus, which is a list of observations
        with known state sequences,
        set the HMM parameters that maximize the corpus likelihood.
        Do not update the transitions if translock is True,
        or the emissions if emitlock is True.
        """
        # TODO: fill in for section b
        pass

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
        pass

# main
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

########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = []
            tag_count = defaultdict(int)
            self.possible_tags = defaultdict(set)
            self.ix_to_tag = defaultdict(str)
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    tagged_word = TaggedWord(token)
                    sentence.append(tagged_word)
                    self.freqDict[tagged_word.word] += 1
                    tag_count[tagged_word.tag] += 1
                    self.possible_tags[tagged_word.word].add(tagged_word.tag)
                sens.append(sentence) # append this list as an element to the list of sentences
            tag_count['<s>'] = len(file)
            self.freqDict = {k: v for k, v in self.freqDict.items() if v >= self.minFreq}
            index = 0
            for word in self.freqDict:
                self.word_to_ix[word] = index
                index += 1            
            index = 0
            self.tag_to_ix['<s>'] = index
            index += 1
            for tag in tag_count:
                if tag == '<s>':
                    continue
                self.tag_to_ix[tag] = index
                self.ix_to_tag[index] = tag
                index += 1
            self.trans_matrix = [[0 for x in tag_count] for y in range(len(tag_count) - 1)]          
            self.emission_matrix = [[0 for x in self.freqDict] for y in tag_count]  
            for sen in sens:
                last = sen[0]
                curr_tag_ix = self.tag_to_ix[last.tag]
                curr_word_ix = self.word_to_ix[last.word]
                self.emission_matrix[curr_word_ix][curr_tag_ix] += (1/tag_count[self.tag_to_ix[last.tag]])
                self.trans_matrix[0][curr_tag_ix] += (1/tag_count['<s>'])
                for i in range(1, len(sen)):
                    last_tag_ix = self.tag_to_ix[last.tag]
                    curr_tag_ix = self.tag_to_ix[sen[i].tag]
                    self.trans_matrix[last_tag_ix][curr_tag_ix] += (1/tag_count[last.tag]) # need to divide by count of last tag
                    curr = sen[i]
                    self.emission_matrix[curr_word_ix][curr_tag_ix] += (1/tag_count[last.tag])
                    last = sen[i]
            
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = []
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        self.freqDict = defaultdict(int)
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.tag_to_ix = {}
        ### Initialize the rest of your data structures here ###

    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile) # data is a nested list of TaggedWords
        #print("Your first task is to train a bigram HMM tagger from an input file of POS-tagged text")

    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    # t_seq = argmax[PI(P(emission)P(transition))]
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        #print("Your second task is to implement the Viterbi algorithm for the HMM tagger")
        # returns the list of Viterbi POS tags (strings)
        lattice = [[0 for x in self.tag_to_ix if x != '<s>'] for y in words] # make sure order of tags is consistent, inner for creates column
        backpointer = [[0 for x in self.tag_to_ix if x != '<s>'] for y in words]
        for tag in self.tag_to_ix:
            if tag == '<s>':
                continue
            tag_index = self.tag_to_ix[tag]
            word_index = self.word_to_ix[words[0]]
            lattice[tag_index][0] = self.trans_matrix[0][tag_index] * self.emission_matrix[word_index][tag_index] # transistion prob from previous to current
            backpointer[tag_index][0] = 0
        for i in range(1, len(words)):
            word_index = self.word_to_ix[words[i]]
            for tag in self.possible_tags[words[i]]: # move down column
                tag_index = self.tag_to_ix[tag]
                prev_max_cell = 0
                prev_max_tag_ix = 0
                for t in range(0, len(self.tag_to_ix) - 1):
                    if lattice[t][i-1] > prev_max_cell:
                        prev_max_cell = lattice[t][i-1]
                        prev_max_tag_ix = t

                lattice[tag_index][i] = self.emission_matrix[word_index][tag_index] * self.trans_matrix[prev_max_tag_ix][tag_index] * prev_max_cell # max transition * previous path prob
                


        return ["NULL"]*len(words) # this returns a dummy list of "NULL", equal in length to words

if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')

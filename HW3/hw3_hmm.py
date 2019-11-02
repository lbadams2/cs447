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
from math import log, exp, inf

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
            line_count = 0
            for line in file:
                line_count += 1
                raw = line.split()
                sentence = []
                for token in raw:
                    tagged_word = TaggedWord(token)
                    sentence.append(tagged_word)
                    self.freqDict[tagged_word.word] += 1
                    tag_count[tagged_word.tag] += 1
                    self.possible_tags[tagged_word.word].add(tagged_word.tag)
                sens.append(sentence) # append this list as an element to the list of sentences
            tag_count['<s>'] = line_count            
            #self.possible_tags[UNK] = set(tag_count.keys()) - {'<s>'}
            unk_tags = set()
            for w in self.freqDict:
                if self.freqDict[w] < 5:
                    unk_tags.update(self.possible_tags[w])
            self.freqDict = {k: v for k, v in self.freqDict.items() if v >= self.minFreq}
            self.possible_tags[UNK] = unk_tags
            self.word_to_ix[UNK] = 0
            index = 1
            for word in self.freqDict:
                self.word_to_ix[word] = index
                index += 1            
            num_tags = len(tag_count)
            self.tag_to_ix['<s>'] = num_tags - 1
            self.ix_to_tag[num_tags - 1] = '<s>'
            index = 0
            for tag in tag_count:
                if tag == '<s>':
                    continue
                self.tag_to_ix[tag] = index
                self.ix_to_tag[index] = tag
                index += 1
            self.trans_matrix = [[0 for x in range(num_tags - 1)] for y in range(num_tags)] # rows include start tag  
            self.emission_matrix = [[0 for x in self.word_to_ix] for y in range(num_tags - 1)]  # tag_count rows by word type columns
            for sen in sens:
                last = sen[0]
                curr_tag_ix = self.tag_to_ix[last.tag]
                if last.word not in self.word_to_ix:
                    curr_word_ix = self.word_to_ix[UNK]
                else:
                    curr_word_ix = self.word_to_ix[last.word]
                self.emission_matrix[curr_tag_ix][curr_word_ix] += 1
                self.trans_matrix[num_tags - 1][curr_tag_ix] += 1
                for i in range(1, len(sen)):
                    last_tag_ix = self.tag_to_ix[last.tag]
                    curr_tag_ix = self.tag_to_ix[sen[i].tag]
                    if sen[i].word not in self.word_to_ix:
                        curr_word_ix = self.word_to_ix[UNK]
                    else:
                        curr_word_ix = self.word_to_ix[sen[i].word]
                    self.trans_matrix[last_tag_ix][curr_tag_ix] += 1
                    self.emission_matrix[curr_tag_ix][curr_word_ix] += 1
                    last = sen[i]

            # smoothing, trans_matrix covers every possible bigram, should be no entries with 0
            # rows should sum to about 1 because count tag in row is used in denominator for probs
            for i in range(num_tags):
                last_tag = self.ix_to_tag[i]
                for j in range(num_tags - 1):
                    self.trans_matrix[i][j] = (self.trans_matrix[i][j] + 1) / (tag_count[last_tag] + num_tags)

            for i in range(num_tags - 1):
                tag = self.ix_to_tag[i]
                for j in range(len(self.word_to_ix)):
                    self.emission_matrix[i][j] = self.emission_matrix[i][j] / tag_count[tag]
            
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
        num_tags = len(self.tag_to_ix)
        lattice = [[0 for x in words] for y in range(num_tags - 1)] # num_tags - 1 rows
        #lattice = [[0 for x in self.tag_to_ix if x != '<s>'] for y in words] # make sure order of tags is consistent, inner for creates column
        backpointer = [[0 for x in words] for y in range(num_tags - 1)]
        #backpointer = [[0 for x in self.tag_to_ix if x != '<s>'] for y in words]   
        if words[0] not in self.word_to_ix:
            word_index = self.word_to_ix[UNK]
        else:
            word_index = self.word_to_ix[words[0]]     
        for tag_index in range(num_tags - 1):            
            trans_prob = 0
            emiss_prob = 0
            if self.trans_matrix[num_tags - 1][tag_index] == 0:
                trans_prob = -inf
            else:
                trans_prob = log(self.trans_matrix[num_tags - 1][tag_index])
            if self.emission_matrix[tag_index][word_index] == 0:
                emiss_prob = -inf
            else:
                emiss_prob = log(self.emission_matrix[tag_index][word_index])
            log_prob = trans_prob + emiss_prob # transistion prob from previous to current
            lattice[tag_index][0] = exp(log_prob)
            backpointer[tag_index][0] = 0
        for i in range(1, len(words)): # move over columns
            last_word = words[i-1]
            if last_word not in self.word_to_ix:
                last_word_index = self.word_to_ix[UNK]
                last_word = UNK
            else:
                last_word_index = self.word_to_ix[words[i-1]]
            word = words[i]
            if word not in self.word_to_ix:
                word_index = self.word_to_ix[UNK]
                word = UNK
            else:
                word_index = self.word_to_ix[words[i]]
            for tag in self.possible_tags[word]: # move down column
                tag_index = self.tag_to_ix[tag]
                max_val = 0
                prev_max_tag_ix = 0
                for t in self.possible_tags[last_word]:
                    prev_tag_ix = self.tag_to_ix[t]
                    trans_prob = 0
                    emiss_prob = 0
                    lat_prob = 0
                    if self.trans_matrix[prev_tag_ix][tag_index] == 0:
                        trans_prob = -inf
                    else:
                        trans_prob = log(self.trans_matrix[prev_tag_ix][tag_index])
                    if self.emission_matrix[tag_index][word_index] == 0:
                        emiss_prob = -inf
                    else:
                        emiss_prob = log(self.emission_matrix[tag_index][word_index])
                    if lattice[prev_tag_ix][i-1] == 0:
                        lat_prob = -inf
                    else:
                        lat_prob = log(lattice[prev_tag_ix][i-1])
                    log_prob = trans_prob + emiss_prob + lat_prob
                    val = exp(log_prob)
                    if val > max_val:
                        max_val = val
                        prev_max_tag_ix = t

                lattice[tag_index][i] = max_val
                backpointer[tag_index][i] = prev_max_tag_ix

        max_prob = 0
        max_tag = 0
        for tag in self.possible_tags[words[-1]]:
            tag_index = self.tag_to_ix[tag]
            if lattice[tag_index][len(words) - 1] > max_prob:
                max_prob = lattice[tag_index][len(words) - 1]
                max_tag = tag_index

        best_path = []
        best_path.insert(0, self.ix_to_tag[max_tag])
        for i in range(len(words) - 2, -1, -1):
            tag_ix = backpointer[max_tag][i]
            tag = self.ix_to_tag[tag_ix]
            best_path.insert(0, tag)

        return best_path

if __name__ == "__main__":
    tagger = HMM()
    #tagger.train('train.txt')
    tagger.train('/Users/liam_adams/my_repos/cs447/HW3/train.txt')
    tagger.test('/Users/liam_adams/my_repos/cs447/HW3/test.txt', 'out.txt')

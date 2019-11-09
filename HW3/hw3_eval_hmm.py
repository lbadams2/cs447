########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict

class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]

# A class for evaluating POS-tagged data
class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        print("Your task is to implement an evaluation program for POS tagging")
        self.gold_token_count = defaultdict(int)
        gold_file = open(goldFile, "r")
        self.gold_tokens = []
        for line in gold_file:
            raw = line.split()
            tokens = []
            for token in raw:
                tagged_word = TaggedWord(token)
                self.gold_token_count[tagged_word.tag] += 1
                tokens.append(tagged_word.tag)
            self.gold_tokens.append(tokens)

        out_file = open(testFile, "r")
        self.out_tokens = []
        for line in out_file:
            raw = line.split()
            tokens = []
            for token in raw:
                tagged_word = TaggedWord(token)
                tokens.append(tagged_word.tag)
            self.out_tokens.append(tokens)

        self.create_confusion_matrix()

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        #print("Return the percentage of correctly-labeled tokens")
        token_count = 0
        correct = 0
        for i in range(len(self.gold_tokens)):
            gold_line = self.gold_tokens[i]
            out_line = self.out_tokens[i]
            if len(gold_line) != len(out_line):
                print("*********** number of tokens in lines not equal **************")
            for j in range(len(gold_line)):
                if gold_line[j] == out_line[j]:
                    correct += 1
                token_count += 1
        return correct / token_count

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        #print("Return the percentage of sentences where every word is correctly labeled")
        correct_sens = 0
        for i in range(len(self.gold_tokens)):
            gold_line = self.gold_tokens[i]
            out_line = self.out_tokens[i]
            if len(gold_line) != len(out_line):
                print("*********** number of tokens in lines not equal **************")
            correct = 0
            for j in range(len(gold_line)):
                if gold_line[j] == out_line[j]:
                    correct += 1
            if correct == len(gold_line):
                correct_sens += 1
        return correct_sens / len(self.gold_tokens)

    def create_confusion_matrix(self):
        self.tag_list = list(self.gold_token_count.keys())        
        self.matrix = [[0 for x in self.tag_list] for y in self.tag_list] # x columns y rows
        # sum of row is number of times tag appears in gold data. Columns break down that count among the predicted tags
        for i in range(len(self.gold_tokens)):
            gold_line = self.gold_tokens[i]
            out_line = self.out_tokens[i]
            for j in range(len(gold_line)):
                g_i = self.tag_list.index(gold_line[j])
                p_j = self.tag_list.index(out_line[j])
                self.matrix[g_i][p_j] += 1

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        #print("Write a confusion matrix to outFile; elements in the matrix can be frequencies (you don't need to normalize)")
        with open(outFile, 'w') as mat_file:
            mat_file.write('\t' + '\t'.join(self.tag_list) + '\n')
            row_ix = 0
            for row in self.matrix:
                cnt_strs = []
                for cnt in row:
                    cnt_strs.append(str(cnt))
                mat_file.write(self.tag_list[row_ix] + '\t' + '\t'.join(cnt_strs) + '\n')
                row_ix += 1



    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):
        #print("Return the tagger's precision when predicting tag t_i")
        tag_ix = self.tag_list.index(tagTi)
        true_pos = self.matrix[tag_ix][tag_ix]
        col_sum = sum(row[tag_ix] for row in self.matrix)
        return true_pos / col_sum
        

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        #print("Return the tagger's recall for correctly predicting gold tag t_j")
        tag_ix = self.tag_list.index(tagTj)
        true_pos = self.matrix[tag_ix][tag_ix]
        row_sum = sum(self.matrix[tag_ix])
        return true_pos / row_sum


if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and test.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print("Token accuracy: ", eval.getTokenAccuracy())        
        print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # Calculate recall and precision
        print("Recall on tag NNP: ", eval.getPrecision('NNP'))
        print("Precision for tag NNP: ", eval.getRecall('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("confusion_matrix.txt")
        
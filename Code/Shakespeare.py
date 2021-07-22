import csv
from nltk import word_tokenize
from nltk import sent_tokenize
import re
import numpy as np


class shakespeare:
    def __init__(self):
        self.file_name = "will_play_text.csv"
        self.playDic = dict()
        self.__load__()
        self.len_chars = 61
        self.chars = " !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


    def __load__(self):
        csvfile = open(self.file_name)
        dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=";,")
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)

        for line in reader:
            if line[1] not in self.playDic:
                self.playDic[line[1]] = list();
            self.playDic[line[1]].append(line[1:]);
        csvfile.close()


    def plays(self):
        '''Return names of all plays'''
        return list(self.playDic.keys())


    def printplays(self):
        '''Print the name of all plays'''
        for key in self.playDic.keys():
            print (key)


    def content(self, play_name):
        '''Given the name of the play print out the entire play'''
        line_num = 1;
        for val in self.playDic[play_name]:

            print(str(line_num) + "\t" + '\t'.join(val))
            line_num += 1


    def script(self, play_name):
        '''Returns the entire script of the play. ONLY THE LINES'''
        lines = str()
        for val in self.playDic[play_name]:
            lines += val[-1]
            lines += " "
        return lines


    def listline(self, play_name):
        '''Return a list where each string is a line'''
        lines = list()
        speaker = ''

        for val in self.playDic[play_name]:
            name = val[-2]
            line = re.sub(r'\[(.*?)\]', '', val[-1])
            line = re.sub(r'\s', ' ', line)

            if speaker == val[-2]:
                lines.append(line)
            else:
                lines.append(name + ': ' + line)
                speaker = name
        return lines


    def charVectors(self, line):
        '''
        input: a line of shakespeare

        output: a dictionary of one-hot vectors of each character at this line
        '''
        cv = {}
        for num, char in enumerate(line):
            vector = np.zeros((self.len_chars, 1))
            vector[self.chars.index(char)] = 1.
            cv[num] = vector
        return cv


    def uniqueChars(self, play_name):
        lines = self.listline(play_name)
        chars = ''
        for line in lines:
            chars = ''.join(sorted(set(chars.join(sorted(set(line))))))
        return chars


    def listword(self, play_name):
        '''return a list of word of the entire script'''
        wordlist = list()
        for val in self.playDic[play_name]:
            for word in word_tokenize(val[-1]):
                "val[-1] are the lines"
                wordlist.append(word)
                "might want to add lower() to avoid duplication"
                "ex Hello != hello; this will lower the chance 'hello' will appear because of case sensitive"
        return wordlist


    def listsent(self, play_name):
        '''return a list of sentence in the play, this concatenate the entire play then call sent_tokenize to
        create a list of sentence'''
        return sent_tokenize(self.script(play_name))

import re
import csv
import nltk
import pandas as pd
from difflib import SequenceMatcher
from collections import OrderedDict
from collections import defaultdict
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from itertools import chain

class similar_word_grouping():
    
    def __init__(self, input_file):
        self.df = pd.read_excel(input_file, index_col=0, engine = 'openpyxl')  
        self.word_list = self.df['cluster_word2pred']
        self.word_dict_list = []
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def word_preprocess(self, word):
        w = ' '.join([j.split('-')[0] for j in word.split('_')]) #remove POS tags
        w = re.sub(r'(.)\1\1+',r'\1\1', w) #remove repeating characters (> 2)
        return w.lower() 
        
    def get_word_dict_list(self):
        word_dict_list = []
        #preprocess -> stem -> lemmatize in order 
        word_dict_list.append([[i, self.word_preprocess(i)] for i in self.word_list])
        word_dict_list.append([[i, self.ps.stem(self.word_preprocess(i))] for i in self.word_list])
        word_dict_list.append([[i, self.lemmatizer.lemmatize(self.word_preprocess(i))] for i in self.word_list])
        return word_dict_list #list of [word with tag, word without tag]
        
    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def find_similar_words(self, threshold, word_dict):
        d = defaultdict(list)
        index = 0
        dup_list = []

        for i in self.word_list: 
            if i not in dup_list: #avoid duplicates
                d[index].append(i)

                w = ' '.join([j.split('-')[0] for j in i.split('_')])

                for j in word_dict:
                    if (i != j[0]) & (self.similar(w, j[1]) > threshold) & (j[0] not in dup_list):
                        d[index].append(j[0])
                        dup_list.append(j[0])
                dup_list.append(i)
                index += 1
                
        return d

    
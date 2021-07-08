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
        word_dict_list.append([[i, self.word_preprocess(i)] for i in self.word_list])
        word_dict_list.append([[i, self.ps.stem(self.word_preprocess(i))] for i in self.word_list])
        word_dict_list.append([[i, self.lemmatizer.lemmatize(self.word_preprocess(i))] for i in self.word_list])
        return word_dict_list #list of [word with tag, word without tag]
        
    
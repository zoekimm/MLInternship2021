import json
from gensim.models.keyedvectors import KeyedVectors
import pickle
import numpy as np
import re

class keywords_embedding:
    
    def __init__(self, jsonfile, txtfile, npyfile):
        self.word_list = self.get_word_list(jsonfile) #words.json
        self.word_embed = self.get_word_embed(npyfile) #word_embeds.npy
        self.word_key = self.get_word_key(txtfile) #vectors.txt
        
    def get_word_list(self, jsonfile):
        with open(jsonfile) as file1: 
            word_list = json.load(file1)
        return word_list
    
    def get_word_embed(self, npyfile):
        return np.load(npyfile)
    
    def get_word_key(self, txtfile):
        model = KeyedVectors.load_word2vec_format(txtfile, binary=False)
        return list(model.index_to_key)
        
    def format_word(self):    
        #self.word_key = [' '.join([j.split('-')[0] for j in i.lower().split('_')]) for i in self.word_key]
    
    
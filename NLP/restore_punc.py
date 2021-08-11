import csv
import spacy
import pandas as pd
from itertools import chain
from fastpunct import FastPunct
from punctuator import Punctuator

class restore_punctuation:
    
    def __init__(self):
        self.text_list = self.get_text_list()
        self.d = self.get_empty_dict()
        
    def get_text_list(self):
        f = open("punctuation_test.txt", "r")
        reader = csv.reader(f)
        text_list = [row for row in reader]
        return list(chain(*text_list))
        
    def get_empty_dict(self):
        return {'text': None, 'fastpunct': None, 'spacy': None, 'INTERSPEECH-T-BRNN.pcl(punctuator)': None, 'Demo-Europarl-EN.pcl(punctuator)': None}
        
    def fastpunct_punct(self, text):
        fastpunct = FastPunct()
        nlp=spacy.load('en_core_web_lg')

        sent=[str(i) for i in nlp(text).sents]
        
        self.d['fastpunct'] = ' '.join(fastpunct.punct(sent, correct=False))

    def spacy_punct(self, text):
        nlp = spacy.load('en_core_web_sm')
        text_sentences = nlp(text)
        li = []
        for sentence in text_sentences.sents:
            li.append(str(sentence.text) + '.')
            
        self.d['spacy'] = ' '.join(li)

    def model1_punct(self, text):
        punctuator1 = Punctuator('INTERSPEECH-T-BRNN.pcl')

        self.d['INTERSPEECH-T-BRNN.pcl(punctuator)'] = punctuator1.punctuate(text)

    def model2_punct(self, text):
        punctuator2 = Punctuator('Demo-Europarl-EN.pcl')

        self.d['Demo-Europarl-EN.pcl(punctuator)'] = punctuator2.punctuate(text)

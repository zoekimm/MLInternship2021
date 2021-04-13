import pickle
import numpy as np
import pandas as pd
import csv
import nltk
import os
from phonetics import metaphone
#from metaphone import doublemetaphone
import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict

class keyword_inputpp:
    
    def __init__(self):
        self.v_lstm = self.get_v_lstm()
        self.c_lstm = self.get_c_lstm()
        self.file_list = []

    def get_v_lstm(self):    
        with open('v_lstm.pickle', 'rb') as pickle2:
            v_lstm = pickle.load(pickle2)
        return v_lstm

    def get_c_lstm(self):
        with open('c_lstm.pickle', 'rb') as pickle1:
            c_lstm = pickle.load(pickle1)
        return c_lstm

    def convert(self, data):
        sentence_list = []
        for x in data:          
            word_list = [metaphone(i) for i in x['sentence'].split(' ')] #metaphone each word
            x['sentence'] = ' '.join(word_list) #join them back to a sentence
            sentence_list.append(x['sentence']) 
        return sentence_list

    def get_file_list(self):
        self.file_list.append(self.convert(self.v_lstm))
        self.file_list.append(self.convert(self.c_lstm))
        
    def load_spm(self, file, file_list, file_name):
        # save it as a text file for sentencepiece
        with open(file_name, mode='wt', encoding='utf-8') as text1:
            text1.write('\n'.join(file_list))

        templates= '--input={} \
                    --pad_id={} \
                    --bos_id={} \
                    --eos_id={} \
                    --unk_id={} \
                    --model_prefix={} \
                    --vocab_size={} \
                    --character_coverage={} \
                    --model_type={}'

        train_input_file = os.path.join(file_name)
        pad_id=0 
        vocab_size = 890 
        prefix = file_name[:-12] + 'spm'
        bos_id=1 
        eos_id=2 
        unk_id=3 
        character_coverage = 1.0
        model_type ='unigram'
        max_sentence_length = 10000

        cmd = templates.format(train_input_file,
                        pad_id,
                        bos_id,
                        eos_id,
                        unk_id,
                        prefix,
                        vocab_size,
                        character_coverage,
                        model_type,
                        max_sentence_length)

        spm.SentencePieceTrainer.Train(cmd)
        sp = spm.SentencePieceProcessor()
        sp.load(file_name[:-12] + 'spm' +'.model')
        for i in file:
             i['sentence'] = sp.encode_as_ids(i['sentence'])
        return 
    
    def load_text(self, filename, data):
        keyword_dict = defaultdict(lambda : len(keyword_dict))

        #write into a text file
        with open(filename, 'w') as outFile:
            for i in data:
                id_arr = ",".join(map(str,i['sentence'])) 
                keyword_arr = []
                for j in i['intended_keywords']:
                    keyword_arr.append(keyword_dict[j])
                keyword_arr = ",".join(map(str,keyword_arr))
                line = id_arr + ' : ' + keyword_arr + '\n'
                outFile.write(line)

        all_ds = tf.data.TextLineDataset(filename)
        return all_ds

    def splitLine(self, line):
        string =tf.strings.split(line,sep=':')
        x = tf.strings.split(string[0],sep=',')
        y = tf.strings.split(string[1],sep=',')
        return x,y  
    
    def sentence_size(self, length):
        def _sentence_size(x, y):
            x = tf.strings.to_number(x, tf.int32)
            x = tf.pad(x,[[0,length-tf.shape(x)[0]]]) #1d 
            y = tf.strings.to_number(y, tf.int32)
            y = tf.pad(y,[[0,length-tf.shape(y)[0]]]) #1d 
            return (y,x), y
        return _sentence_size

    def execute(self):
        
        self.get_file_list()
        
        self.load_spm(self.v_lstm, self.file_list[0], 'v_lstm_sentence.txt')
        self.load_spm(self.c_lstm, self.file_list[1], 'c_lstm_sentence.txt') 

        v_ds = self.load_text('v_lstm_encoded.txt', self.v_lstm) 
        v_ds = v_ds.repeat().shuffle(10000).map(self.splitLine).map(self.sentence_size(128))

        c_ds = self.load_text('c_lstm_encoded.txt', self.c_lstm) 
        c_ds = c_ds.repeat().shuffle(10000).map(self.splitLine).map(self.sentence_size(128))

def main():
    # generate model class
    inputpp = keyword_inputpp()
    
    # execute model
    inputpp.execute()

if __name__ == '__main__':
    main()
    


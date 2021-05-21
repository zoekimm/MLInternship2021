import pickle
import os
import sys
import sentencepiece as spm
from phonetics import metaphone
from collections import defaultdict
from sentencepiece import sentencepiece_model_pb2 as model
import tensorflow as tf

class sentence_tfdata:
  
    def __init__(self):
        self.f = self.get_file()
        
    def get_file(self):
        with open('voice.pickle', 'rb') as pickle2:
            voice_lstm = pickle.load(pickle2)
        with open('comments.pickle', 'rb') as pickle1:
            comments_lstm = pickle.load(pickle1) 
        f = []
        for i in voice_lstm:
            f.append(i)
        for j in comments_lstm:
            f.append(j)
        return f
      
    def load_text(self, filename, data):
        sp = spm.SentencePieceProcessor()
        sp.load('spm3_modified.model')
        
        #write into a text file
        with open(filename, 'w') as outFile:
            for i in data[0:2]:
                id_arr = ",".join(map(str, sp.encode_as_ids(i['sentence']))) 
                a = list(map(lambda x: x.rstrip(), i['intended_keytalks']))
                y = self.convert_onehot(self.remove_b(i['intended_keytalks']))
                line = id_arr + ' : '
                outFile.write(line)
                for element in y:
                    outFile.write(str(element) + ' ')
                outFile.write('\n')
        all_ds = tf.data.TextLineDataset(filename)
        return all_ds
      
    def remove_b(self, li):
        t = []
        for i in li:
            t.append(i.rstrip())
        return t 
      
    def convert_onehot(self, y):
        with open('word2idx_gen.pickle', 'rb') as pickle2:
            new_y_word2idx_gen = pickle.load(pickle2)
        new_y_word2idx_gen2 = {k.rstrip():v for k,v in new_y_word2idx_gen.items()}
        y = list(map(lambda y: new_y_word2idx_gen2[y], y))
        y = tf.one_hot(y, depth = len(new_y_word2idx_gen))  
        y = tf.reduce_sum(y, axis = 0) 
        return y.numpy()
      
    def splitLine(self, line):
        string = tf.strings.split(line,sep=':')
        x = tf.strings.split(string[0],sep=',')
        y = tf.strings.split(string[1],sep=' ')
        return x,y  
      
    def sentence_size(self, length):
        def _sentence_size(x, y):
            x = tf.strings.to_number(x, tf.int32) #encode as id -> spm model
            #x = tf.pad(x,[[0,length-tf.shape(x)[0]]]) #1d 
            y = tf.strings.to_number(y, tf.int32)
            return (y,x), y
        return _sentence_size
      
    def execute(self):
        ds = self.load_text('sentence2.txt', self.f) 
        ds = ds.shuffle(10000).map(self.splitLine).map(self.sentence_size(128))
        #save as tfrecord

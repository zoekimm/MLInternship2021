import pickle
import os
import sys
import sentencepiece as spm
from phonetics import metaphone
from collections import defaultdict
from sentencepiece import sentencepiece_model_pb2 as model
import tensorflow as tf

class sentence_tfdata:
  
    def __init__(self, model, pickle1, pickle2):
        self.f = self.get_file(pickle1, pickle2)
        self.sp = self.get_sp(model)
        self.d = []
        
    def get_sp(self, model):
        sp = spm.SentencePieceProcessor()
        sp.load(model)
        return sp

    def get_file(self, file1, file2):
    
        with open(file1, 'rb') as pickle1:
            voice_lstm = pickle.load(pickle1)

        with open(file2, 'rb') as pickle2:
            comments_lstm = pickle.load(pickle2) 
            
        f = []
            
        for i in voice_lstm:
            f.append(i)

        for j in comments_lstm:
            f.append(j)
            
        return f
        
    def create_dict(self):
        
        for i in self.f:
            self.d.append({'sentence': i['sentence'], 'intended_keywords': i['intended_keywords']})

            meta_li = []
            for n in range(5):
                meta_li.append(self.sp.encode(metaphone(i['sentence']), out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))
            
            for j in meta_li:
                self.d.append({'sentence': ' '.join(j), 'intended_keywords': i['intended_keywords']})
        
    def convert_onehot(self, y):
    
        #words to index pickle file
        with open('word2idx.pickle', 'rb') as pickle2:
            new_y_word2idx_gen = pickle.load(pickle2)

        new_y_word2idx_gen2 = {k.rstrip():v for k,v in new_y_word2idx_gen.items()}
        y = list(map(lambda y: new_y_word2idx_gen2[y], y))
        y = tf.one_hot(y, depth = len(new_y_word2idx_gen))  
        y = tf.reduce_sum(y, axis = 0)  #14096
        return y.numpy()
    
    def remove_b(self, li):
        #remove unnecessary spaces 
        return [x.rstrip() for x in li]
    
    def create_dataset(self):
            
        X = list(map(lambda x :self.zero_pad(x['sentence'], 32), self.d))

        Y = list(map(lambda x :self.convert_onehot(self.remove_b(x['intended_keywords'])), self.d))
        
        return tf.data.Dataset.from_tensor_slices(((Y, X), Y))
        
    def encode_id(self, i, length):
        i = self.sp.encode(i)
        i = tf.pad(i, [[0, length - len(i)]]) #1d 
        i = tf.cast(i, dtype = tf.int32)
        return i
               
    def __call__(self):
        self.create_dict()
        ds = self.create_dataset()
        for line in ds:
            print(line)
            break

    def zero_pad(self, i, X_seq_len = 32):
        input_ids = [j + 1 for j in self.sp.encode(i)]
        if len(input_ids) < X_seq_len:
            n_pad_tokens = X_seq_len - len(input_ids)
            input_ids.extend([0] * n_pad_tokens)
            return input_ids
        return input_ids[:32]

def main():
    # generate model class
    inputpp = sentence_tfinputpp('voice_spm3_modified.model','voice.pickle', 'comments.pickle')
    
    # execute model
    em = inputpp()

if __name__ == '__main__':
    main()
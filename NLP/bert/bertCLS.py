#keyword(1) similarity weight(all 1) tl_name
import pickle
from transformers import BertTokenizer, TFBertModel
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import os
from tempfile import mkdtemp
import os.path as path


class bert_CLS:
    
    def __init__(self, shape):
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.mat = self.get_empty_mat(shape)

    def get_tokenizer(self):
        return BertTokenizer.from_pretrained("bert-base-uncased")
    
    def get_model(self):
        return TFBertModel.from_pretrained('bert-base-uncased')
    
    def get_empty_mat(self, shape):
        filename = path.join(mkdtemp(), 'word_list.dat')
        print(filename)
        return np.memmap(filename, dtype='float32', mode='w+', shape = shape) 
        
    def tokenize(self, st):
        st = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(st))
        st = [self.tokenizer.vocab[self.tokenizer.cls_token]] + st
        st = np.expand_dims(st, axis = 0)
        return st
    
    def __call__(self, input_file):
        df = pd.read_csv(input_file, delimiter = "\n", names = ['st'])
        for index, row in tqdm(df.iterrows(), total = df.shape[0]):
            st = row['st']
            st = self.tokenize(st) 
            y = self.model.predict(st)
            arr = np.expand_dims(np.squeeze(y[1]), axis = 0)
            self.mat[index] = arr[:]
    
def main(input_file, shape):
    #input_file : file name
    #shape : dimension
    
    # generate model class 
    model = bert_CLS(shape)  
    
    # execute model
    em = model(input_file)
    return em

if __name__ == '__main__':
    input_file = 'vocab_list.txt'
    shape = (24946, 768)
    result = main(input_file, shape)
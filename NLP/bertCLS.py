import pickle
from transformers import BertTokenizer, TFBertModel
import numpy as np

class bert_CLS:
    def __init__(self, st):
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.st = st
        
    def get_tokenizer(self):
        return BertTokenizer.from_pretrained("bert-base-uncased")
      
    def get_model(self):
        return TFBertModel.from_pretrained('bert-base-uncased')
      
    def tokenize(self):
        self.st = "[CLS]" + str(self.st)
        self.st = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.st))
        self.st = [self.st]
        
    def execute(self):
        self.tokenize() 
        y = self.model.predict(self.st)
        y = np.expand_dims(y, axis = 0)
        return y[0][1]
      
def main(input_string):
    # generate model class 
    st = [input_string]
    inputpp = bert_CLS(st)  
    # execute model
    mat = inputpp.execute()

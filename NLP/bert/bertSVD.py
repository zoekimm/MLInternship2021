import pickle
from transformers import BertTokenizer, TFBertModel
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.sparse import linalg as lg
import os.path as path

class bert_svd:
    
    def __init__(self, mat, dim):
        self.mat = self.get_svd(mat, dim)
        self.cosmat = self.get_cosmat()
        
    def get_svd(self, mat, dim):
        return (lg.svds(mat, dim))[0]
        
    def get_cosmat(self):
        return 1-pairwise_distances(self.mat, metric="cosine")
        
    def create_dic(self, input_file):
        df = pd.read_csv(input_file, delimiter = "\n", names = ['st'])

        d_list = []
        weight = np.full((np.shape(self.cosmat)[0], ), 1)

        for index, row in tqdm(df.iterrows(), total = df.shape[0]):
            d = {'keywords' : row['st'],'similarity' : self.cosmat[index], 'weight' : weight}
            d_list.append(d)

        return d_list

    def extract_top(self, d_list):
        new_list = []
        weight = np.full((20, ), 1)

        for i in tqdm(range(0, (np.shape(self.cosmat)[0]))):
            top_sim = np.sort(d_list[i]['similarity'])[::-1][:20]
            largest_indices = (np.argsort(d_list[i]['similarity']))[::-1][:20]

            keyword_list = []
            for i in largest_indices:
                keyword_list.append(d_list[i]['keywords'])

            d = {'keywords' : keyword_list,'similarity' : top_sim, 'weight' : weight}
            new_list.append(d)

        return new_list

    def to_pickle(self, new_list, pickle_name):
        with open(pickle_name, 'wb') as f:
            pickle.dump(new_list, f)
  
    def __call__(self, input_file, pickle_name):
            self.get_cosmat()
            d_list = self.create_dic(input_file)
            new_list = self.extract_top(d_list)
            self.to_pickle(new_list, pickle_name)

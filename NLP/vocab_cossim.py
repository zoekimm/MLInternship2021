from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def open_memmap(filename):
    #open original memmap 
    vocab_memmap = np.memmap(filename, dtype='float32', mode='r')
    vocab_memmap = vocab_memmap.reshape((36723, 768))
    return vocab_memmap
  
def get_cosmat(mat):
    return 1-pairwise_distances(mat, metric="cosine")
  
def create_dic(cosmat):
    df = pd.read_csv('vocab_list.txt', delimiter = "\n", names = ['st'])
    d_list = []
    weight = np.full((36723, ), 1)
    
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        d = {'keywords' : row['st'].split(' '),'similarity' : cosmat[index], 'weight' : weight}
        d_list.append(d)
        
    return d_list

def extract_top(d_list):
    new_list = []
    weight = np.full((20, ), 1)
    
    for i in tqdm(range(0, 36723)):
        arr = d_list[i]['similarity']
        top_sim = np.sort(arr)[::-1][:20]
        ranked = np.argsort(d_list[i]['similarity'])
        largest_indices = ranked[::-1][:20]
        keyword_list = []
        
        for i in largest_indices:
            keyword_list.append(d_list[i]['keywords'])
        d = {'keywords' : keyword_list,'similarity' : top_sim, 'weight' : weight}
        new_list.append(d)
        
    return new_list

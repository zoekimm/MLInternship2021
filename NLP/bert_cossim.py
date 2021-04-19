from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def open_memmap():
    vocab_memmap = np.memmap(filename, dtype='float32', mode='r')
    vocab_memmap = vocab_memmap.reshape((36723,768))
    return vocab_memmap
  
def get_cosmat(mat):
    return 1-pairwise_distances(mat, metric="cosine")
  
def create_dic(cosmat):
    df = pd.read_csv('vocab_list_new.txt', delimiter = "\n", names = ['st'])
    d_list = []
    weight = np.full((36723, ), 1)
    
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        d = {'keywords' : row['st'].split(' '),'similarity' : cosmat[index], 'weight' : weight}
        d_list.append(d)
        
    return d_list
  
def main():
    mat = open_memmap()
    cosmat = get_cosmat(mat)
    d_list = create_dic(cosmat)
    
    with open('vocab.pickle', 'wb') as f:
        pickle.dump(d_list, f)

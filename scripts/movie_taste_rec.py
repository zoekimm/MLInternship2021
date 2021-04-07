from scipy.sparse import load_npz
from scipy.sparse import linalg as lg
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json
import csv

def normsk(x):
    return normalize(x, norm='l1', axis=1) #normalize using sklearn 

def concat(x, y):
    #normalize beforehand
    row = x[0].shape[0]
    col = x[0].shape[1] + y[0].shape[1]
    mat = np.zeros((row, col), dtype=np.float64)
    mat[:, :x[0].shape[1]] = x[0]
    mat[:, x[0].shape[1]:] = y[0]
    return mat 

def svd2(x, y):
    concat_list = []
    sm = [8, 16, 32, 64]
    x = lg.svds(x, 128)
    for i in sm:
        concat_list.append(concat(x, lg.svds(y, i)))
    return concat_list

def cossim(x, df, movie_dict):
    #cossim(x, df, movie_dict, movie_list):
    pos = 1
    for i in x: 
        colname = df.columns[pos]
        r = 0
        for j in range(1, i.shape[0]) #for all movies 
        #for j in movie_list: # certain movies 
            r = r + 1
            arr = np.dot(i[j], i.T)/np.linalg.norm(i, axis = 1)
            ten_indices = np.argsort(-1*arr)[:11]
            ten_movies = []
            for k in ten_indices: #return titles
                ten_movies.append(movie_dict[str(k)])
            ten_movies.pop(0) #delete duplicates
            df.at[r, colname] = ten_movies
        pos = pos + 1 
    return df
    
def main():
    mat1 = load_npz('movie.npz')
    mat2 = load_npz('movie2.npz')
    with open('movieList.json') as file:
        movie_dict = json.load(file)
    mat1 = normsk(mat1)
    mat2 = normsk(mat2)
    mat_list = svd2(mat1, mat2)

    COLUMN_NAMES = ['movie', '8', '16', '32', '64']
    df = pd.DataFrame(index=range(1, len(movie_list) + 1), columns=COLUMN_NAMES)
    row_num = 1

    for i in movie_list:
            df.at[row_num, 'movie'] = movie_dict[str(i)]
            row_num = row_num + 1
        
    #result_df = cossim(mat_list, df, movie_dict, movie_list)
    result_df = cossim(mat_list, df, movie_list)
    result_df.to_csv('movie_taste_recommendation_list.csv')

from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans
import pickle
import numpy as np
from collections import defaultdict
from itertools import chain
import umap.umap_ as umap
import hdbscan

class vectors_cluster:
    def __init__(self, filename):
        self.model = KeyedVectors.load_word2vec_format(filename, binary = False)
        self.vectors = self.get_vectors()
        self.indices = self.get_indices()
        self.threshold = int(self.vectors.shape[0] * 0.2)
        self.size = 0
        self.label = {}
        self.d = defaultdict(list)
        self.clusters = {}
        
    def get_vectors(self):
        return np.asarray(self.model.vectors)
      
    def get_indices(self):
        return np.asarray(self.model.index_to_key)
      
    def __call__(self):
        self.get_clusters()
        self.count_label()
        with open('cluster_combined_dict.pickle', 'wb') as f:
            pickle.dump(self.d, f)
            
    def get_clusters(self):
        self.size = int(self.vectors.shape[0])
        clusterable_embedding = umap.UMAP(n_neighbors = 10,
                                        min_dist = 0.1,
                                        n_components = 64,random_state = 42,).fit_transform(self.vectors)
        self.label = hdbscan.HDBSCAN(min_cluster_size = 10, min_samples = 5).fit_predict(clusterable_embedding)
        with open('cluster_hdbscan.pickle', 'wb') as f:
            pickle.dump(self.label, f)
            
    def count_label(self):
        for i, x in enumerate(self.label):
            self.d[x].append(i)
            
        for k, v in self.d.items():
            self.d[k] = np.take(self.indices, self.d[k], axis = 0)
            
        if (len(self.d[-1]) > self.threshold):
            print(len(self.d[-1]))
            self.recluster()
            
    def recluster(self):
        reword_list = list(self.d[-1])
        self.indices = list(self.indices)
        locations = [self.indices.index(i) for i in reword_list]
        self.vectors = np.take(self.vectors, locations, axis = 0)
        self.indices = np.take(self.indices, locations, axis = 0)
        
        clusterable_embedding = umap.UMAP(n_neighbors = 10,
                                        min_dist = 0.1,
                                        n_components = 64,random_state = 42,).fit_transform(self.vectors)
        
        self.label = hdbscan.HDBSCAN(min_cluster_size = 10, min_samples = 5).fit_predict(clusterable_embedding)
        update_key = defaultdict(list)
        
        for i, x in enumerate(self.label):
            update_key[x].append(i)
            
        for k, v in update_key.items():
            update_key[k] = np.take(self.indices, update_key[k], axis = 0)
            
        self.size = len(self.d)
        del self.d[-1]
        for k, v in update_key.items():
            self.d[k + self.size] = v
     

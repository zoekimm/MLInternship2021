#hdbscan -> recluster 80% hdbscan -> reculster kmeans

from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
        self.size = int(self.vectors.shape[0])
        self.label = {}
        self.d = defaultdict(list)
        self.clusters = {}
        self.n_neighbors = 10
        self.min_dist = 0.1
        self.min_cluster_size = 20
        self.min_samples = 5
        
    def get_vectors(self):
        return np.asarray(self.model.vectors)
    
    def get_indices(self):
        return np.asarray(self.model.index_to_key)
    
    def __call__(self):
        self.get_clusters()
        self.make_dict(self.d)

        open_file = open('cluster_1.pickle', "rb")
        self.d = pickle.load(open_file)
        
        self.count_label()
   
        with open('cluster_3.pickle', 'wb') as f:
            pickle.dump(self.d, f)
        
    def get_clusters(self):
        clusterable_embedding = umap.UMAP(n_neighbors = self.n_neighbors,
                                        min_dist = self.min_dist,
                                        n_components = 64, random_state = 42,).fit_transform(self.vectors)
        self.label = hdbscan.HDBSCAN(min_cluster_size = self.min_cluster_size, min_samples = self.min_samples).fit_predict(clusterable_embedding)
           
    def make_dict(self, d):
        
        for i, x in enumerate(self.label):
            d[x].append(i)

        for k, v in self.d.items():
            d[k] = np.take(self.indices, d[k], axis = 0)
    
    def count_label(self):
        
        if (len(self.d[-1]) > self.threshold):
            print(len(self.d[-1]))
            #self.recluster()
        else:
            if (len(self.d[-1]) != 0):
                print("recluster kmeans")
                self.get_kmeans(self.d[-1])
            
    
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from sklearn.cluster import KMeans
import pickle
import numpy as np

class vectors_cluster:
  
    def __init__(self, filename):
        self.model = KeyedVectors.load_word2vec_format(filename, binary=False)
        self.vectors = self.get_vectors()
        self.indices = self.get_indices()
        
    def get_vectors(self):
        return np.asarray(self.model.wv.vectors)
      
    def get_indices(self):
        return {word: self.model.wv.vocab[word].index for word in self.model.wv.vocab}
      
    def reduce_dim(self):
        pca = PCA(n_components = 64)
        self.vectors = pca.fit_transform(self.vectors) #reduce dimension
        
    def __call__(self):     
        kmeans_clustering = KMeans(n_clusters = int(self.vectors.shape[0]/10))
        idx = kmeans_clustering.fit_predict(self.vectors)
        words = self.model.wv.index2word
        word_centroid_map = {words[i]: idx[i] for i in range(len(words))}
        return word_centroid_map

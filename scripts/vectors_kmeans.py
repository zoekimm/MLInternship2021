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
        self.word_map1 = {}
        self.word_map2 = {}
        
    def get_vectors(self):
        return np.asarray(self.model.wv.vectors)
      
    def get_indices(self):
        return {word: self.model.wv.vocab[word].index for word in self.model.wv.vocab}
      
    def reduce_dim(self):
        pca = PCA(n_components = 64)
        self.vectors = pca.fit_transform(self.vectors) #reduce dimension
        
   def __call__(self):
        self.reduce_dim()
        self.get_clusters()
        self.recluster()
        self.combine_cluster()
        
    def get_clusters(self):
        #hdbscan
        kmeans_clustering = KMeans(n_clusters = int(self.vectors.shape[0]/10))
        idx = kmeans_clustering.fit_predict(self.vectors)
        words = self.model.wv.index2word
        self.word_map1 = {words[i]: idx[i] for i in range(len(words))}
        with open('cluster1.pickle', 'wb') as f:
            pickle.dump(self.word_map1, f)
            
    def get_new_word(self):  
        new_dict = defaultdict(list)
        reword_list = []
        
        for k, v in self.word_map1.items():
            new_dict[v].append(k) 
            
        for k, v in new_dict.items():  
            if ((len(v) >= 30) | (len(v) <= 3)):
                reword_list.append(new_dict[k])
                
        return list(chain(*reword_list))
      
    def recluster(self):
        reword_list = self.get_new_word()
        locations = [self.indices[i] for i in reword_list]
        new_vector = np.take(vectors, locations, axis = 0)
        kmeans_clustering = KMeans(n_clusters = int(new_vector.shape[0]/10))
        idx = kmeans_clustering.fit_predict(new_vector)
        words = np.take(self.model.wv.index2word, locations, axis = 0)
        self.word_map2 = {words[i]: idx[i] for i in range(len(words))}
        
        with open('cluster2.pickle', 'wb') as f:
            pickle.dump(self.word_map2, f)
            
    def combine_cluster(self):
        for k, v in self.word_map1.items():
            if k in self.word_map2:
                self.word_map1[k] = self.word_map2[k] + int(self.vectors.shape[0]/10)
                print(self.word_map1[k], self_word_map2[k])
                
        with open('cluster_combined.pickle', 'wb') as f:
            pickle.dump(self.word_map1, f)

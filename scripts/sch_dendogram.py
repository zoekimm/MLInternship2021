from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

class h_dendrogram:
    def __init__(self, array):
        self.model = self.get_model(array)
    def get_model(self, array):
        return hierarchy.linkage(array, 'complete')
    def __call__(self):
        fig = plt.figure()
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('index')
        plt.ylabel('distance')
        hierarchy.dendrogram(self.model, leaf_font_size = 8, leaf_rotation= 0)
        
def main(array):  
    # generate model class 
    model = h_dendrogram(array)  
    ex = model()
   

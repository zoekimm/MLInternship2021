from sklearn import preprocessing
from scipy import stats
import numpy as np

class zscore_outlier():
    def __init__(self, inputlist):
        self.outliers = []
        self.og = []
        self.median = 0
        self.median_deviation = 0
        self.inputlist = inputlist
        self.list = self.get_list(inputlist)
        
    def min_max_norm(self, inputlist):
        arr = np.array(inputlist)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        inputlist = arr.tolist()
        for _ in range(0, len(self.outliers)):
            inputlist.append(1)  
        return inputlist
      
    def find_outliers(self, inputlist):
        non_outliers = [i for i in inputlist if i < 3 or i < -3]
        self.outliers = list(set(inputlist) - set(non_outliers))
        return non_outliers
      
    def outliers_modified_z_score(self, threshold=3):
        arr = np.array(self.inputlist)
        self.median = np.median(arr)
        self.median_deviation = np.median(np.abs(arr - self.median))
        z_score = 0.7 * (arr - self.median) / self.median_deviation
        return z_score
      
    def get_list(self, inputlist):
        x = self.outliers_modified_z_score(inputlist)
        x = self.find_outliers(x)
        self.og = np.array(x)
        x = self.min_max_norm(x)
        return x 
      
    def __call__(self, val):
        val_z = 0.7 * (val - self.median) / self.median_deviation
        val = (val_z - self.og.min()) / (self.og.max() - self.og.min())
        if np.array(self.list).max() < val:
            self.list.append(1)
        else:
            self.list.append(val)
        return self.list
 

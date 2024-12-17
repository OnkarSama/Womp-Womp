import pandas as pd
import numpy as np
class VectorKNNClassifier:

    def __init__(self, data = pd.DataFrame, k_neighbors : int = 3, weights : list = None):
        self.k_neighbors = k_neighbors

        if(weights == None):
            self.weights = np.ones(len(data.columns))
        else:
            self.weights = weights

        self.data = data

    def show(self):
        print("k number of neighbors:", self.k_neighbors)
        print("The Weights being used:", self.weights)
        print("The Data being used:", self.data)

    # def
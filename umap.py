import numpy as np
from pynndescent import NNDescent

class UMAP():
    def __init__(self, n_neighbors=10, dims=2, min_dist=.1, epochs=10):
        """
        Parameters
        -----------
        n_neighbors : int
            neigborhood size
        dims : int
            dimension of target space
        min_dist : float
            parameter controlling layout
        epochs : int
            amount of optimization to perform
        """
        self.n_neighbors = n_neighbors
        self.dims = dims
        self.min_dist = min_dist
        self.epochs = epochs
        self.knn_dists = None
        self.knn_indices = None
    
    def get_neighbors(self, X):
        """
        Get k nearest neighbors for all points using nearest neighbor descent
        """
        index = NNDescent(X, n_neighbors=self.n_neighbors)
        self.knn_indices, self.knn_dists = index.neighbor_graph

    def spectralEmbedding(self, top_reps):
        pass

    def optimizeEmbedding(self, top_rep, Y):
        pass
    
    def smoothKNNDist(self, distances, rho):
        """
        Binary search for sigma such that the sum of all neighbors ...  = log2(n)
        
        Parameters
        -----------
        rho : float
            distance to nearest neighbor

        Returns
        --------
        sigma : float
            smooth approximator to knn-distance
        """
        mid = 1.0
        lo = 0.0
        hi = np.inf
        k = len(distances)
        target = np.log2(k)
        
        #64 iteratopms
        for n in range(64):
            psum = 0.0
            #go through distances to all neighbors
            for j in range(1, distances.shape(1)):
                d = distances[j] - rho 
                if d > 0:
                    psum += np.exp(-(d/mid))
                else:
                    psum += 1.0
            
            #if sum is close enough to log2(k)
            if np.fabs(psum - np.log2(k)):
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            
            else:
                lo = mid
                if hi == np.inf:
                    mid *=2
                else:
                    mid = (lo + hi) /2.0

        #maybe need to add scaling
        return mid

    def localFuzzySimplicialSet(self, X,x):
        """
        Parameters
        -----------
        X : array of shape (n_samples, n_features)
            dataset
        x : int
            index of X
        
        Returns
        --------
        fs_set : ?

        """
        rho = self.knn_dists[x][1]
        sigma = self.smoothKNNDist(self.knn_dists[x], rho)
        pass

    def fit(self, X):
        """
        Parameters
        -----------
        X : array of shape (n_samples, n_features)
            dataset
        """
        #construct relevant wieghted graph
        #for all x in X, use local fuzzy simplicalset(X,x,n)
        fs_set = []
        for x in range(X.shape[0]):
            fs_set[x] = self.localFuzzySimplicialSet(X,x)
        #use probabilitic t-conorm?
        top_rep = None

        #perform optimization of the graph layout
        Y = self.spectralEmbedding(top_rep)
        Y = self.optimizeEmbedding(top_rep, Y)
    
    
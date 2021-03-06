import numpy as np
from pynndescent import NNDescent
import umap
import scipy
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import curve_fit
import stochastic_gradient_descent

"""Implementation of UMAP"""

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
        self.Y = None
        self.a = 1
        self.b = 1
        self.graph_cols = None
        self.graph_rows = None
    
    def get_neighbors(self, X):
        """
        Get k nearest neighbors for all points using nearest neighbor descent

        Parameters
        ----------
        X : array (n_samples, n_features)
        """
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        index = NNDescent(X, metric='euclidean',n_neighbors=self.n_neighbors, low_memory=False, n_jobs=-1, verbose=True, n_iters=n_iters, n_trees=n_trees, max_candidates=60)
        self.knn_indices, self.knn_dists = index.neighbor_graph

    def spectralEmbedding(self, graph):
        """
        Computes embedding of graph into specified dimensions
        parameters
        Adapted from official UMAP implementation
        ---------
        graph : coo_matrix
            adjacency matrix of graph
        
        Returns
        --------
        embedding : array
    
        """
        diag_data = np.asarray(graph.sum(axis=0))
        I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
        D = scipy.sparse.spdiags(
            1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
        )
        L = I - D * graph * D
        k = self.dims + 1
        random_state = check_random_state(8)
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
        
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]

    def optimizeEmbedding(self, top_rep, Y):
        """
        Stochastic gradient descent optimization
        
        Parameters
        ----------
        top_rep : matrix
            fuzzy set graph
        Y : array
            initial embedding
        """

        return stochastic_gradient_descent.optimize_embedding(Y,Y,self.a,self.b, self.dims, self.epochs, top_rep, self.graph_rows, self.graph_cols)
        
    def find_ab_params(self, spread, min_dist):
        """Fit a, b params for the differentiable curve 
        Adapted from official UMAP implementation

        Parameters
        ----------
        spread : float
        min_dist : float
        """

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, covar = curve_fit(curve, xv, yv)
        return params[0], params[1]

    def smoothKNNDist(self, distances, rho):
        """
        Binary search for sigma such that the sum of all neighbors ...  = log2(n)
        Adapted from UMAP implementation

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
        
        #64 iterations
        for i in range(64):
            psum = 0.0
            #go through distances to all neighbors
            for j in range(1, k):
                d = distances[j] - rho 
                if d > 0:
                    psum += np.exp(-(d/mid))
                else:
                    psum += 1.0
            
            #if sum is close enough to log2(k)
            if np.fabs(psum - target) < 1e-5:
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
        MIN_K_DIST_SCALE = 1e-3
        if rho > 0.0:
            mean_ith_distances = np.mean(distances)
            if mid < MIN_K_DIST_SCALE * mean_ith_distances:
                mid = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            mean_distances = np.mean(self.knn_dists)
            if mid < MIN_K_DIST_SCALE * mean_distances:
                mid = MIN_K_DIST_SCALE * mean_distances
        return mid

    def computeSkeletalEdges(self, indices, dists, rho, sigma):
        """
        Get weight of edges between vertex and it's neighbors

        Parameters
        ----------
        indices : array
        dists : array
        rho : float
        sigma : float

        Returns
        -------
        edges : array
            [((x,y),weight)]
            edges computed for 1-skeleton

        """
        edges = []
        x = indices[0]
        for index, coord in enumerate(indices):
            if coord == x: 
                edges.append(((x,coord), 0.0))
                continue
            if dists[index] - rho < 0 or sigma == 0:
                d_xy = 1
            else:
                d_xy = np.exp(-((dists[index] - rho) / sigma))
            
            edges.append(((x,coord), d_xy))
        
        return edges

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
        fs_set : array
            [((x,y),weight)]

        """
        rho = self.knn_dists[x][1]
        sigma = self.smoothKNNDist(self.knn_dists[x], rho)
        edges = self.computeSkeletalEdges(self.knn_indices[x], self.knn_dists[x], rho, sigma)
        
        return edges

    def combine_sets(self, X, fs_set):
        """
        Parameters
        ----------
        fs_set : array 
            fuzzy simplicial sets for each point
        
        Returns
        ----------
        graph : coo_matrix
            adjacency matrix representing combined fuzzy simplicial sets
        """
        rows = []
        cols = []
        weights = []
        #combine 1-skeletons
        for one_skeleton in fs_set:
            for edge in one_skeleton:
                rows.append(edge[0][0])
                cols.append(edge[0][1])
                weights.append(edge[1])
            
        graph = scipy.sparse.coo_matrix((weights, (rows, cols)), shape=(X.shape[0], X.shape[0]))
        graph.eliminate_zeros()
        self.graph_rows = graph.row
        self.graph_cols = graph.col
        """transpose = graph.transpose()
        prod_matrix = graph.multiply(transpose)
        graph = (
            (graph + transpose - prod_matrix)
        )
        graph.eliminate_zeros()"""
        #graph = graph.tocoo()
        #graph.sum_duplicates()
        #graph = graph + np.transpose(graph) - np.multiply(graph, np.transpose(graph))
        return graph

    def fit(self, X, use_precomputed=False):
        """
        Finds lower dimensional embedding
        
        Parameters
        -----------
        X : array of shape (n_samples, n_features)
            dataset
        use_precomputed : Boolean
            True if precomputed neighbors should be looked for
        """
        
        if use_precomputed:
            try:
                self.knn_indices = np.loadtxt('k_neighbors/' + str(self.n_neighbors) + '_indices.csv', delimiter=",")
                self.knn_dists = np.loadtxt('k_neighbors/' + str(self.n_neighbors) + '_dists.csv', delimiter=",")
            except OSError:
                print("getting neighbors")
                self.get_neighbors(X)
                print("done getting neighbors")
        else: 
            print("getting neighbors")
            self.get_neighbors(X)
            print("done getting neighbors")

        self.a, self.b = self.find_ab_params(1, self.min_dist)
        
        #construct relevant wieghted graph
        #for all x in X, use local fuzzy simplicalset(X,x,n)
        fs_set = []

        for x in range(X.shape[0]):
            fs_set.append(self.localFuzzySimplicialSet(X,x))
        
        top_rep = self.combine_sets(X, fs_set)

        #perform optimization of the graph layout
        self.Y = self.spectralEmbedding(top_rep)
        self.Y = self.optimizeEmbedding(top_rep, self.Y)
    
    
    
import numpy as np
from pynndescent import NNDescent
import scipy
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances

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
    
    def get_neighbors(self, X):
        """
        Get k nearest neighbors for all points using nearest neighbor descent
        """
        index = NNDescent(X, n_neighbors=self.n_neighbors)
        self.knn_indices, self.knn_dists = index.neighbor_graph

    def spectralEmbedding(self, graph):
        """
        Computes embedding of graph into specified dimensions
        parameters
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
        """eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                L, random_state.normal(size=(L.shape[0], k)), largest=False, tol=1e-8
            )"""
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]

    def optimizeEmbedding(self, top_rep, Y):
        """
        Stochastic gradient descent optimization
        right now not stochastic
        """
        a=self.a
        b=self.b
        alpha = 1.0
        for i in range(self.epochs):
            print('starting ' + str(i) + ' epoch')
            
            for x in range(top_rep.toarray().shape[0]):
                for y in range(top_rep.toarray().shape[1]):
                    #print(str(x) + "," + str(y))
                #if np.random.randint() < top_rep[1]:
                    
                    dist, dist_grad = self.euclidean_grad(Y[x], Y[y])
                    if dist > 0.0:
                        w_l = pow((1 + a * pow(dist, 2 * b)), -1)
                    else:
                        w_l = 1.0
                    grad_coeff = 2 * b * (w_l - 1) / (dist + 1e-6)

                    for d in range(self.dims):
                        grad_d = grad_coeff * dist_grad[d]
                        #print(grad_d)
                        Y[x][d] += alpha * grad_d
                    
                    for p in range(10):
                        #k = tau_rand_int(rng_state) % n_vertices
                        k = np.random.randint(len(Y))
                        current = Y[x]
                        other = Y[k]

                        dist_output, grad_dist_output = self.euclidean_grad(
                            current, other
                        )

                        if dist_output > 0.0:
                            w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                        elif x == k:
                            continue
                        else:
                            w_l = 1.0
                        gamma = 1.0
                        grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

                        for d in range(self.dims):
                            grad_d = grad_coeff * grad_dist_output[d]
                            Y[x][d] += grad_d * alpha

            alpha = alpha * (1.0 - (float(i) / float(self.epochs)))
            print('Done with ' + str(i) + ' epoch')
        return Y
        """
        for i in range(self.epochs):
            Y = Y - self.CE_gradient(top_rep, Y)

        return Y
        """
    
    def euclidean_grad(self, x, y):
        """Standard euclidean distance and its gradient.
        ..math::
            D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
            \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
        """
        result = 0.0
        for i in range(x.shape[0]):
            result += (x[i] - y[i]) ** 2
        d = np.sqrt(result)
        grad = (x - y) / (1e-6 + d)
        return d, grad

    def prob_low_dim(self, Y):
        """
        Compute matrix of probabilities q_ij in low-dimensional space
        """
        inv_distances = np.power(1 + self.a * np.square(euclidean_distances(Y, Y))**self.b, -1)
        return inv_distances

    def CE(self, P, Y):
        """
        Compute Cross-Entropy (CE) from matrix of high-dimensional probabilities 
        and coordinates of low-dimensional embeddings
        """
        Q = self.prob_low_dim(Y)
        return - P * np.log(Q + 0.01) - (1 - P) * np.log(1 - Q + 0.01)

    def CE_gradient(self, P, Y):
        """
        Compute the gradient of Cross-Entropy (CE)
        """
        P = np.matrix(P.toarray())
        Y = np.array(Y)
        y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        inv_dist = np.power(1 + self.a * np.square(euclidean_distances(Y, Y))**self.b, -1)
        Q = np.dot(1 - P, np.power(0.001 + np.square(euclidean_distances(Y, Y)), -1))
        np.fill_diagonal(Q, 0)
        #Q = np.asarray(Q)
        Q = Q / np.sum(Q, axis = 1, keepdims = True)
        fact=np.expand_dims(self.a*P*(1e-8 + np.square(euclidean_distances(Y, Y)))**(self.b-1) - Q, 2)
        return 2 * self.b * np.sum(fact * y_diff * np.expand_dims(inv_dist, 2), axis = 1)

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

    def computeSkeletalEdges(self, indices, dists, rho, sigma):
        """
        Parameters
        ----------
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
        for y in range(len(indices) - 1):
            if dists[y] - rho < 0 or sigma == 0:
                d_xy = 1.0
            else:
                d_xy = (dists[y] - rho) / sigma
            
            edges.append(((x,y), np.exp(-d_xy)))
        
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
        graph.sum_duplicates()
        return graph

    def fit(self, X):
        """
        Parameters
        -----------
        X : array of shape (n_samples, n_features)
            dataset
        """
        #construct relevant wieghted graph
        #for all x in X, use local fuzzy simplicalset(X,x,n)
        self.get_neighbors(X)
        fs_set = []
        for x in range(X.shape[0]):
            fs_set.append(self.localFuzzySimplicialSet(X,x))
        
        top_rep = self.combine_sets(X, fs_set)
        print(str(len(top_rep.row)))
        print(str(len(top_rep.col)))
        print(top_rep.toarray())
        #perform optimization of the graph layout
        self.Y = self.spectralEmbedding(top_rep)
        print(len(self.Y))
        self.Y = self.optimizeEmbedding(top_rep, self.Y)
    
    
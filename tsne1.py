import numpy as np

class TSNE():


    def getPH(self, D, sigma = 1.0):
        """
        Calculate all P(i) and H(P(i)) given a specific sigma 
        Parameters
        -----------
        D : np.array([n][d])
            array with n datapoints and d dimentions
        sigma : float
            Gaussian variable for individual vertice
        """
        P = np.exp(-D.copy() * sigma)
        sumP = sum(P)
        H = np.log(sumP) + sigma * np.sum(D * P) / sumP
        P = P / sumP
        return P, H 

    def assignPValues(self, X, perplexity = 30.0, tolerance = 1e-5):
        """
        Instantiate all the p(i|j) values for data
        Parameters
        -----------
        X : np.array([n][d])
            array with n datapoints and d dimentions
        perplexity : int
            used to get sigma for distribution probability
        tolerance : float
            how acurate the binary search is
        """
        #Equation 1 from t-SNE paper
        (n, d) = X.shape
        P = np.zeros((n,n))
        sigma = np.ones((n,1))
        sum_X = np.sum(np.square(X), 1)
        #Adapted from t-SNE implementation
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        target = np.log(perplexity)

        #Find P for every node
        for i in range(n):

            smin = -np.inf
            smax = np.inf

            #r_ combines along the first axis, removes i from list
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]

            #get first H(pi) for sigma of 1
            (Pi, H) = self.getPH(Di, sigma[i])

            j = 0
            max_attempts = 50
            Haway = H - target

            #Binary search for sigma
            while np.abs(Haway) > tolerance and max_attempts > j:
                if Haway > 0:
                    smin = sigma[i].copy()
                    if smax == np.inf or smax == -np.inf:
                        sigma[i] = sigma[i] * 2.
                    else:
                        sigma[i] = (sigma[i] + smax) / 2.
                else:
                    smax = sigma[i].copy()
                    if smin == np.inf or smin == -np.inf:
                        sigma[i] = sigma[i] / 2.
                    else:
                        sigma[i] = (sigma[i] + smin) / 2.
                (Pi, H) = self.getPH(Di, sigma[i])
                Haway = H - target
                j += 1
            #insert Pi into the offical P for all
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pi

        return P

    def tsne(self, X, dims = 2, perplexity = 20, exageration = 2.,
         momentum = 0.65, iterations = 200):
        """
        Parameters
        -----------
        X : np.array([n][d])
            array with n datapoints and d dimentions, representing the data in high dimention
        dims : int
            dimension of lower dimention array
        perplexity : int
            used to get sigma for distribution probability
        exageration : float
            how quickly to accelerate the change at the start
        momentum : float
            amount of change to apply to the gradient
        iteration : int
            amount of iterations to change the lower dimentions
        """

        #Random initial solution Y(0) = {y1, y2, ... yn} from N(0, 10^-4I)
        #Y is the final output
        (n, d) = X.shape
        Y = np.random.rand(n, dims)
        gradient = np.zeros((n, dims))
        Yfirst = np.zeros((n, dims))
        Ysecond = np.zeros((n, dims))
        gains = np.ones((n, dims))
        min_gain = 0.01
        eta = 500


        #compute pairwise affinities p(i|j) with perplexity Perp 
        #Equation 1 from t-SNE paper
        P = self.assignPValues(X, perplexity, 1e-5)
        #set p(ij) = p(j|i) + p(i|j) / 2n
        P = P + np.transpose(P)
        P = P / (2*np.sum(P))
        P = P * float(exageration)

        
        #compute Q(i|j) t(iteration) times
        for t in range(iterations):

            #compute low dimensional affinities q(ij) 
            #Equation 4 from t-SNE paper
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.

            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            #compute gradient dC/dY EQ5
            PminQ = P - Q
            for i in range(n):
                gradient[i, :] = np.sum(np.tile(PminQ[:, i] * num[:, i], (dims, 1)).T * (Y[i, :] - Y), 0)

            #set Y(t) = Y(t-1) + lr(C/Y) + a(t)(Y(t-1) - Y(t-2))
            gains = (gains + 0.2) * ((gradient > 0.) != (Yfirst > 0.)) + \
                    (gains * 0.8) * ((gradient > 0.) == (Yfirst > 0.))
            gains[gains < min_gain] = min_gain
            Yfirst = momentum * Yfirst - eta * (gains * gradient)
            Y = Y + Yfirst
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        return Y
        


import numpy as np
import pylab
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import penguins as pg


class TSNE():
        

    def __init__(self, iterations = 200, learning_rate = 0.01, momentum = 0.65):
        self.iterations = iterations
        self.lr = learning_rate
        self.momentum = momentum


    def getPH(self, D, beta = 1.0):
        P = np.exp(-D.copy() * beta)
        sumP = sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return P, H 

    def assignPValues(self, X, perplexity = 30.0, tolerance = 1e-5):

        (n, d) = X.shape
        P = np.zeros((n,n))
        beta = np.ones((n,1))
        #no clue what this does
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        target = np.log(perplexity)

        #Find P for every node
        for i in range(n):
            if i % 100 == 0:
                print("Computing P-values for point %d of %d..." % (i, n))
            bmin = -np.inf
            bmax = np.inf

            #r_ combines along the first axis, removes i from list
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            #get the H(pi) for i 
            (Pi, H) = self.getPH(Di, beta[i])

            j = 0
            max_attempts = 50
            Haway = H - target
            while np.abs(Haway) > tolerance and max_attempts > j:
                if Haway > 0:
                    bmin = beta[i].copy()
                    if bmax == np.inf or bmax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + bmax) / 2.
                else:
                    bmax = beta[i].copy()
                    if bmin == np.inf or bmin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + bmin) / 2.
                (Pi, H) = self.getPH(Di, beta[i])
                Haway = H - target
                j += 1

            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pi

        return P

    def pca(self, X=np.array([]), no_dims=50):
        """
            Runs PCA on the NxD array X in order to reduce its dimensionality to
            no_dims dimensions.
        """

        print("Preprocessing the data using PCA...")

        (n, d) = X.shape
        X = X - np.tile(np.mean(X, 0), (n, 1))
        (l, M) = np.linalg.eig(np.dot(X.T, X))
        Y = np.dot(X, M[:, 0:no_dims])
        return Y

    def tsne(self, X, dims = 2, perplexity = 20, next_dims = 50, exageration = 2.):
        #PCA to get a lower dimention to 50 if needed

        #get the high dimentional shape
        X = self.pca(X, next_dims).real
        (n, d) = X.shape

        #sample initial solution Y(0) = {y1, y2, ... yn} from N(0, 10^-4I)
        #Y is the final output
        Y = np.random.rand(n, dims)
        gradient = np.zeros((n, dims))
        Yfirst = np.zeros((n, dims))
        Ysecond = np.zeros((n, dims))
        gains = np.ones((n, dims))
        min_gain = 0.01
        eta = 500


        #compute pairwise affinities p(i|j) with perplexity Perp 
        # EQ1 from paper
        P = self.assignPValues(X, perplexity, 1e-5)
        #set p(ij) = p(j|i) + p(i|j) / 2n
        P = P + np.transpose(P)
        P = P / (2*np.sum(P))
        P = P * float(exageration)								# early exaggeration

        
        #for t = 1 to iteration do 
        for t in range(self.iterations):

            if t % 50 == 0:
                print("Iteration %d of %d..." % (t, self.iterations))

            #compute low dimensional affinities q(ij) 
            # EQ4 from paper
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.

            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            #compute gradient C/Y EQ5
            PminQ = P - Q
            for i in range(n):
                gradient[i, :] = np.sum(np.tile(PminQ[:, i] * num[:, i], (dims, 1)).T * (Y[i, :] - Y), 0)

            #set Y(t) = Y(t-1) + lr(C/Y) + a(t)(Y(t-1) - Y(t-2))
            gains = (gains + 0.2) * ((gradient > 0.) != (Yfirst > 0.)) + \
                    (gains * 0.8) * ((gradient > 0.) == (Yfirst > 0.))
            gains[gains < min_gain] = min_gain
            Yfirst = self.momentum * Yfirst - eta * (gains * gradient)
            Y = Y + Yfirst
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))
            if iter == 100:
                P = P / float(exageration)
        return Y
        

            
        return Y
if __name__ == "__main__":
    tsne = TSNE(iterations=200)

    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")

    penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
    penguins = penguins.dropna()
    penguin_data = penguins[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    ].values
    scaled_penguin_data = StandardScaler().fit_transform(penguin_data)

    X = scaled_penguin_data
    Y = tsne.tsne(X, 2, 20.0, exageration=4)

    labels = penguins[
    [
        "species_short",
    ]
    ].values
    f_label = []
    for i in range(len(labels)):
        if labels[i] == 'Adelie':
            f_label.append(1)
            #labels[i] = 1
        elif labels[i] == 'Chinstrap':
            #labels[i] = 2
            f_label.append(2)
        else: 
            #labels[i] = 3
            f_label.append(3)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, f_label)
    pylab.show()
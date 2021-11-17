#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')
import pylab
import experiment_helpers
from contrastive_loss import ContrastiveLoss



class DiffusionMap():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def find_diffusion_matrix(self, X=None, alpha=0.15):
        """Function to find the diffusion matrix P
            
         >Parameters:
         alpha - to be used for gaussian kernel function
         X - feature matrix as numpy array
            
         >Returns:
         P_prime, P, Di, K, D_left
         """
        alpha = alpha
        print('alpha value:', alpha)
        dists = euclidean_distances(X, X)
        K = np.exp(-dists**2 / alpha)
        
        r = np.sum(K, axis=0)
        Di = np.diag(1/r)
        P = np.matmul(Di, K)
        
        D_right = np.diag((r)**0.5)
        D_left = np.diag((r)**-0.5)
        P_prime = np.matmul(D_right, np.matmul(P,D_left))
    
        return P_prime, P, Di, K, D_left
    
    def find_diffusion_map(self, P_prime, D_left, n_eign=3):
             
        n_eign = n_eign
        
        eigenValues, eigenVectors = eigh(P_prime)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        diffusion_coordinates = np.matmul(D_left, eigenVectors)
        
        return diffusion_coordinates[:,:n_eign]
     
    
    def plot_diffusion(self, d_map, title='Diffused points'):
       
        pylab.scatter(d_map[:,0], d_map[:,1], c = self.labels)        
        pylab.title(title)
        pylab.show()
        

    def apply_diffusions(self, data, alpha_value):
        d_maps = []
        P_prime, P, Di, K, D_left = self.find_diffusion_matrix(self.data, alpha_value)
        d_maps.append(self.find_diffusion_map(P_prime, D_left, n_eign=2))
        return d_maps[0]
    

if __name__ == '__main__':   
    loss = ContrastiveLoss()
    
    training_data,training_labels,testing_data,testing_labels = experiment_helpers.get_Data()
    diff_map = DiffusionMap(training_data, training_labels)
    
    curr_loss =loss.get_loss(diff_map.data, diff_map.labels)
    print('pre-diffusion loss ', curr_loss)
    
    #Start of diffusion
    d_maps_final = diff_map.apply_diffusions(training_data, .8)
    diff_map.plot_diffusion(d_maps_final)
    
    curr_loss =loss.get_loss(d_maps_final, diff_map.labels)
    print('post-diffusion loss ', curr_loss)
    
    
        
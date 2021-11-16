#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pylab

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

penguin_labels = penguins[
    [
        "species_short"
    ]
    ].values 


#scaled_penguin_data = StandardScaler().fit_transform(penguin_data)

training_data, testing_data = train_test_split(penguin_data, test_size=0.2, random_state=35)
training_labels, testing_labels = train_test_split(penguin_labels, test_size=0.2, random_state=35)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


# print(training_data)
# print(training_labels)
#if isinstance(training_data,(np.ndarray)):
        #print('yay!')
        
#diffusion kernel = exp(-((euclid dist**2))/alpha)
#alpha function set to 0.15 in implementation
def find_diffusion_matrix(X=None, alpha=0.15):
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
    
def find_diffusion_map(P_prime, D_left, n_eign=3):
         
    n_eign = n_eign
    
    eigenValues, eigenVectors = eigh(P_prime)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    diffusion_coordinates = np.matmul(D_left, eigenVectors)
    
    return diffusion_coordinates[:,:n_eign]
    

def plot_diffusion(d_map, title='Diffused points'):
    f_label = []
    for i in range(len(training_labels)):
        if penguin_labels[i] == 'Adelie':
            f_label.append(1)
            #labels[i] = 1
        elif penguin_labels[i] == 'Chinstrap':
            #labels[i] = 2
            f_label.append(2)
        else:
            #labels[i] = 3
            f_label.append(3)
            
            
    pylab.scatter(d_map[:,0], d_map[:,1], c=f_label)        
    pylab.title(title)
    pylab.show()
    
def apply_diffusions_training(alpha_value):
    d_maps = []
    #alpha_values = np.linspace(alpha_start, alpha_end, 10)
    # for alpha in alpha_values:
    #      P_prime, P, Di, K, D_left = find_diffusion_matrix(training_data, alpha=alpha)
    #      d_maps.append(find_diffusion_map(P_prime, D_left, n_eign=2))
    P_prime, P, Di, K, D_left = find_diffusion_matrix(training_data, alpha_value)
    d_maps.append(find_diffusion_map(P_prime, D_left, n_eign=2))
    #print(d_maps[0])
    return d_maps[0]

def apply_diffusions(data, alpha_value):
    d_maps = []
    P_prime, P, Di, K, D_left = find_diffusion_matrix(data, alpha_value)
    d_maps.append(find_diffusion_map(P_prime, D_left, n_eign=2))
    return d_maps[0]

#d_maps = apply_diffusions_training(4)
#plot_diffusion(d_maps)

d_maps = apply_diffusions(training_data,15)
plot_diffusion(d_maps)

    
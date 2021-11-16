import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from contrastive_loss import ContrastiveLoss
import umap
import math

"""Script with helper methods for experiments"""

def get_Data():
    """
    reads in data and split 80/20 for training and testing

    Returns
    ---------
    train_data : array
    train_label :array
    test_data : array
    test_label : array
    """
    penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
    penguins = penguins.dropna()
    penguin_data = penguins[
    [
        "species_short",
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    ].values
    
    for i in range(len(penguin_data)):
        name = penguin_data[i][0]
        if name == 'Adelie':
            penguin_data[i][0] = 1
        elif name == 'Chinstrap':
            penguin_data[i][0] = 2
        else: 
            penguin_data[i][0] = 3
            
    train, test = splitData(penguin_data)
    
    train_data = [data[1:] for data in train]
    train_data = StandardScaler().fit_transform(train_data)
    test_data = [data[1:] for data in test]
    test_data = StandardScaler().fit_transform(test_data)
    train_label = [data[0] for data in train]
    test_label = [data[0] for data in test]
    return train_data, train_label, test_data, test_label

def splitData(data):
    """splits data 80/20"""
    np.random.shuffle(data)
    split = int(len(data)*0.8)
    train, test = np.split(data, [split])
    return train, test

def tsne_coordinate_descent(data, label, tsne):
    """coordinate descent method for TSNE"""
    score = np.inf
    n_score = 0
    iterations = 100
    tol = 100
    
    perplexity = 30
    exageration = 4
    momentum = 0.65
    while abs(score - n_score) > tol:
        n_score = score
        p, n = get_best_perplexity(data, label, exageration, momentum, iterations, tsne)
        print(perplexity, n)
        if n < score:
            score = n
            perplexity = p 
        
        e, n = get_best_exageration(data, label, perplexity, momentum, iterations, tsne)
        print(exageration, n)
        if n < score:
            score = n
            exageration = e
        
        m, n = get_best_momentum(data, label, perplexity, exageration, iterations, tsne)
        print(momentum, n)
        if n < score:
            score = n
            momentum = m
        
        print()
    
def get_best_perplexity(data, label, e, m, i, tsne):
    """GSS for perplexity parameter"""
    loss = ContrastiveLoss()
    a = 10
    b = 60
    gr = (np.sqrt(5)+1)/2
    tolerance = 5
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    min_loss = 0
    count = 0
    while abs(b - a) > tolerance:
        #print(count, c, d)
        Y = tsne.tsne(data, dims = 2, perplexity = c, exageration = e, momentum = m, iterations = i)
        c_loss = loss.get_loss(Y, label)
        
        Y = tsne.tsne(data, dims = 2, perplexity = d, exageration = e, momentum = m, iterations = i)
        d_loss = loss.get_loss(Y, label)
        if c_loss < d_loss:
            b = d
            min_loss = c_loss
        else:
            a = c
            min_loss = d_loss
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        count += 1
    return (b + a) / 2, min_loss

def get_best_exageration(data, label, p, m, i, tsne):
    """GSS for exageration parameter"""
    loss = ContrastiveLoss()
    a = 1
    b = 8
    gr = (np.sqrt(5)+1)/2
    tolerance = 1
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    min_loss = 0
    count = 0
    while abs(b - a) > tolerance:
        #print(count, c, d)
        Y = tsne.tsne(data, dims = 2, perplexity = p, exageration = c, momentum = m, iterations = i)
        c_loss = loss.get_loss(Y, label)
        
        Y = tsne.tsne(data, dims = 2, perplexity = p, exageration = d, momentum = m, iterations = i)
        d_loss = loss.get_loss(Y, label)
        if c_loss < d_loss:
            b = d
            min_loss = c_loss
        else:
            a = c
            min_loss = d_loss
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        count += 1
    return (b + a) / 2, min_loss

def get_best_momentum(data, label, p, e, i, tsne):
    """GSS for momentum parameter"""
    loss = ContrastiveLoss()
    a = 0.2
    b = 1
    gr = (np.sqrt(5)+1)/2
    tolerance = 0.05
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    min_loss = 0
    count = 0
    while abs(b - a) > tolerance:
        #print(count, c, d)
        Y = tsne.tsne(data, dims = 2, perplexity = p, exageration = e, momentum = c, iterations = i)
        c_loss = loss.get_loss(Y, label)
        
        Y = tsne.tsne(data, dims = 2, perplexity = p, exageration = e, momentum = d, iterations = i)
        d_loss = loss.get_loss(Y, label)
        if c_loss < d_loss:
            b = d
            min_loss = c_loss
        else:
            a = c
            min_loss = d_loss
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        count += 1
    return (b + a) / 2, min_loss
    
def umap_coordinate_descent(data, label):
    """coordinate descent method for UMAP"""
    score = np.inf
    n_score = 0
    iterations = 100
    tol = 100
    
    epochs = 200
    neighbors = 15
    min_dist = 0.1

    while abs(score - n_score) > tol:
        n_score = score
        e, n = get_best_epochs(data, label, epochs, neighbors, min_dist)
        print(e, n)
        if n < score:
            score = n
            epochs = e

        neigh, n = get_best_neighbors(data, label, epochs, neighbors, min_dist)
        print(neigh, n)
        if n < score:
            score = n
            neighbors = neigh
        
        m, n = get_best_min_dist(data, label, epochs, neighbors, min_dist)
        print(m, n)
        if n < score:
            score = n
            min_dist = m
        
        print(n_score)
        print(score)
        print()
    
    print("Epochs: " + str(epochs))
    print("Neighbors: " + str(neighbors))
    print("min-dist: " + str(min_dist))
    return (epochs, neighbors, min_dist, score)

def get_best_epochs(data, label, epochs, neighbors, min_dist):
    """GSS for epochs parameter"""
    loss = ContrastiveLoss()
    a = 0
    b = 2000
    gr = (np.sqrt(5)+1)/2
    tolerance = 50
    
    c = math.floor(b - (b - a) / gr)
    d = math.floor(a + (b - a) / gr)
    min_loss = 0
    count = 0
    while abs(b - a) > tolerance:
        #print(count, c, d)
        reducer = umap.UMAP(n_neighbors=neighbors, n_epochs=c, verbose=False, min_dist=min_dist)
        embedding = reducer.fit_transform(data)
        c_loss = loss.get_loss(embedding, label)
        
        reducer = umap.UMAP(n_neighbors=neighbors, n_epochs=d, verbose=False, min_dist= min_dist)
        embedding = reducer.fit_transform(data)
        d_loss = loss.get_loss(embedding, label)

        if c_loss < d_loss:
            b = d
            min_loss = c_loss
        else:
            a = c
            min_loss = d_loss
        c = math.floor(b - (b - a) / gr)
        d = math.floor(a + (b - a) / gr)
        count += 1
    return math.floor((b + a) / 2), min_loss

def get_best_neighbors(data, label, epochs, neighbors, min_dist):
    """GSS for epochs parameter"""
    loss = ContrastiveLoss()
    a = 5
    b = 50
    gr = (np.sqrt(5)+1)/2
    tolerance = 5
    
    c = math.floor(b - (b - a) / gr)
    d = math.floor(a + (b - a) / gr)
    min_loss = 0
    count = 0
    while abs(b - a) > tolerance:
        #print(count, c, d)
        reducer = umap.UMAP(n_neighbors=c, n_epochs=epochs, verbose=False, min_dist=min_dist)
        embedding = reducer.fit_transform(data)
        c_loss = loss.get_loss(embedding, label)
        
        reducer = umap.UMAP(n_neighbors=d, n_epochs=epochs, verbose=False, min_dist= min_dist)
        embedding = reducer.fit_transform(data)
        d_loss = loss.get_loss(embedding, label)
        
        if c_loss < d_loss:
            b = d
            min_loss = c_loss
        else:
            a = c
            min_loss = d_loss
        c = math.floor(b - (b - a) / gr)
        d = math.floor(a + (b - a) / gr)
        count += 1
    return math.floor((b + a) / 2), min_loss

def get_best_min_dist(data, label, epochs, neighbors, min_dist):
    """GSS for epochs parameter"""
    loss = ContrastiveLoss()
    a = .01
    b = .99
    gr = (np.sqrt(5)+1)/2
    tolerance = 0.05
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    min_loss = 0
    count = 0
    while abs(b - a) > tolerance:
        #print(count, c, d)
        reducer = umap.UMAP(n_neighbors=neighbors, n_epochs=epochs, verbose=False, min_dist=c)
        embedding = reducer.fit_transform(data)
        c_loss = loss.get_loss(embedding, label)
        
        reducer = umap.UMAP(n_neighbors=neighbors, n_epochs=epochs, verbose=False, min_dist= d)
        embedding = reducer.fit_transform(data)
        d_loss = loss.get_loss(embedding, label)
        
        if c_loss < d_loss:
            b = d
            min_loss = c_loss
        else:
            a = c
            min_loss = d_loss
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        count += 1
    return (b + a) / 2, min_loss
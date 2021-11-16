from my_umap import UMAP
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns

"""script for running umap implementation for testing"""

if __name__ == '__main__':
    myUmap = UMAP(n_neighbors = 30, dims=2, min_dist=.1, epochs=1000)
    penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
    penguins = penguins.dropna()
    #print(penguins.head())
    penguin_data = penguins[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    ].values
    scaled_penguin_data = StandardScaler().fit_transform(penguin_data)
    
    test = np.array([[1,1,1],[1.1,1.1,1.1],[.9,.9,.9],[4.9,4.9,4.9],[5,5,5]])
    myUmap.fit(scaled_penguin_data)
    to_save_indices = np.array(myUmap.knn_indices)
    to_save_dists = np.array(myUmap.knn_dists)
    np.savetxt(('/Users/daniel/desktop/cp307/dimensional-reduction/k_neighbors/' + str(myUmap.n_neighbors) + '_indices.csv'), to_save_indices, delimiter=',')
    np.savetxt(('/Users/daniel/desktop/cp307/dimensional-reduction/k_neighbors/' + str(myUmap.n_neighbors) + '_dists.csv'), to_save_dists, delimiter=',')
    """reducer = umap.UMAP(n_neighbors=15, n_epochs=0, verbose=True)
    embedding = reducer.fit(scaled_penguin_data)
    print(embedding.shape)
    plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
    
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Penguin dataset', fontsize=24)
    plt.show()"""
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
    """print(len(myUmap.Y))
    print()
    print()
    print(myUmap.Y)"""
    plt.scatter(myUmap.Y[:, 0], myUmap.Y[:, 1], 20, f_label)
    plt.show()
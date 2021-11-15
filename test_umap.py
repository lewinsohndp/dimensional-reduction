from my_umap import UMAP
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
if __name__ == '__main__':
    myUmap = UMAP(n_neighbors = 15, dims=2, min_dist=.1, epochs=10)
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
    test = np.array([[1,1,1,1,1,1,1],[2,2,2,2,2,2,2],[3,3,3,3,3,3,3],[4,4,4,4,4,4,4],[5,5,5,5,5,5,5]])
    myUmap.fit(scaled_penguin_data)
    #reducer = umap.UMAP()
    #reducer.fit(scaled_penguin_data)

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
    print(len(myUmap.Y))
    print(myUmap.Y)
    plt.scatter(myUmap.Y[:, 0], myUmap.Y[:, 1], 20, f_label)
    plt.show()
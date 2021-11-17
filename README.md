# dimensional-reduction
Repository for CP307 Data Structures and Algorithms Final Project

UMAP:
To use UMAP class instantiate my_umap.UMAP with wanted paramters. Then call UMAP.fit(X) where X is the dataset to be embedded in low dimension. The embedding can be accessed with UMAP.Y.

t-SEN:
To use the t-SNE Class, create an instace. To run a t-SNE on a dataset, call the tsne method and pass in the array of data. All parameters are set as inputs for the tsne method. It then returns an array of length of the input with a lower dimention. 

Diffusion Map: To use the DiffusionMap class, create an instance (for example one called diff_map) of the class and fill in needed parameters (data and labels). Using the DiffusionMap instance, call the apply_difffusions method, passing in the instance's data variable (ex: diff_map.data) and a specified alpha variable (default is .8). Save what is returned from the apply_diffusions method as a variable. Then using the instance of the DiffusionMap class, call the plot_diffusions method, passing in the returned value from apply_diffusions.

UMAP Files:
1. my_umap.py: Main script for UMAP implemtation containing UMAP Class.
2. test_umap.py: Script for basic tests of UMAP
3. stochastic_gradient_descent.py: Script containing method to optimize low dimensional embedding
4. umap.ipynb: Jupyter Notebook containing various initial experiments on offical and our UMAP implementation
5. k_neighbors: directory containing some precomputed KNNs for penguin dataset.
6. example_umap.ipynb: basic example of how to use UMAP implementation on Penguin data

tSNE files:
1. tsne1.py: Whole script for the t-SNE implementation containing the TSNE Class 
2. tsne.ipynb: Jupyter Notebook that has multiple experiments and test, some are also in experiments.ipynb

Diffusion Mapping Files:
1. diffusion_map.py: Script for diffusion map which contains the DiffusionMap class.
2. DiffMapExperiments.ipynb: Jupiter Notebook of different experiments using a pydiffmap Diffusion Map to test how changing alpha and kneighbors values impact quality and runtime. Also found in experiments.ipynb

Experiments:
For each algorithm we performed measured contrastive loss as a function of different parameters. For UMAP we tested epochs vs. contrastive loss for both our implementation and the official implementation. For tSNE we tested iterations vs contrastive loss and perplexity vs. loss. For diffusion maps we tested K neighbors vs contrastive loss and alpha vs contrastive loss. We then compared the algorithms by splitting the data 80/20 for training and validation. We then found the optimal parameters for each algorithm using coordinate descent and then tested the algorithms on the validation set. We plotted algorithm vs quality and algorithmn vs time for this test.

1. experiments.ipynb: Jupyter Notebook containing all official experiments.
2. experiments_helpers.py: Script containing methods to help with experiments. Mainly coordinate descent algorithms.
3. contrastive_loss.py: Script containing class to calculate contrastive loss of embedding.

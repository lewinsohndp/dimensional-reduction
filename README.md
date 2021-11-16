# dimensional-reduction
Repository for CP307 Data Structures and Algorithms Final Project

UMAP:
To use UMAP class instantiate my_umap.UMAP with wanted paramters. Then call UMAP.fit(X) where X is the dataset to be embedded in low dimension. The embedding can be accessed with UMAP.Y.

UMAP Files:
1. my_umap.py: Main script for UMAP implemtation containing UMAP Class.
2. test_umap.py: Script for basic tests of UMAP
3. stochastic_gradient_descent.py: Script containing method to optimize low dimensional embedding
4. umap.ipynb: Jupyter Notebook containing various initial experiments on offical and our UMAP implementation
5. k_neighbors: directory containing some precomputed KNNs for penguin dataset.

tSNE files:
Diffusion Mapping Files:

Experiments:
1. experiments.ipynb: Jupyter Notebook containing all official experiments.
2. experiments_helpers.py: Script containing methods to help with experiments. Mainly coordinate descent algorithms.
3. contrastive_loss.py: Script containing class to calculate contrastive loss of embedding.

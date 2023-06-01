# Cluster-Analysis-of-scRNA-seq-data

This is my class project. This project is inspired from graph-sc model [1]. Graph-sc modeled scRNA-seq data as a gene-to-cell graph and utilized a graph auto encoder to learn the cell embeddings in an unsupervised way. Before feeding the gene-to-cell graph into the autoencoder, they reduced the dimension of the features of the gene node to 50 feature spaces using Principal Component Analysis (PCA). In this project, I experimented graph-sc model with several non-linear dimensionality reduction techniques such as: ISOMAP, MDS (Multidimensional Scaling) to observer differences in scRNA-seq clusetering results.

Datasets: https://github.com/xuebaliang/scziDesk/tree/master/dataset

Codes are also inspired from following github repisitories:
1. https://github.com/xuebaliang/scziDesk
2. https://github.com/ciortanmadalina/graph-sc

References:
1: Ciortan M, Defrance M. GNN-based embedding for clustering scRNA-seq data. Bioinformatics. 2022 Feb 15;38(4):1037-44.

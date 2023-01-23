import os
import pickle
import time
from collections import Counter

import dgl
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             calinski_harabasz_score,
                             normalized_mutual_info_score, silhouette_score)
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from tqdm.notebook import tqdm


def evaluate(model, dataloader, n_clusters, plot=False, save=False, cluster=["KMeans"], use_cpu=False, cluster_params ={}):
    """
    Test the graph autoencoder model.

    Args:
        model ([type]): [description]
        dataloader ([type]): [description]
        n_clusters ([type]): [description]
        plot (bool, optional): [description]. Defaults to False.
        save (bool, optional): [description]. Defaults to False.
        cluster (list, optional): [description]. Defaults to ["KMeans"].
        use_cpu (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    device = get_device(use_cpu=use_cpu)
    model.eval()
    z = []
    y = []
    order = []  # the dataloader shuffles samples
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        adj_logits, emb = model.forward(blocks, input_features)
        z.extend(emb.detach().cpu().numpy())
        if "label" in blocks[-1].dstdata:
            y.extend(blocks[-1].dstdata["label"].cpu().numpy())
        order.extend(blocks[-1].dstdata["order"].cpu().numpy())

    z = np.array(z)
    y = np.array(y)
    order = np.array(order)
    order = np.argsort(order)
    z = z[order]
    y = y[order]
    if pd.isnull(y[0]):
        y = None

    k_start = time.time()
    scores = {"ae_end": k_start}
    if save:
        scores["features"] = z
        scores["y"] = y[order] if y is not None else None

    if "KMeans" in cluster:
        kmeans = KMeans(n_clusters=n_clusters,
                        init="k-means++", random_state=5)
        kmeans_pred = kmeans.fit_predict(z)
        ari_k = None
        nmi_k = None
        if y is not None:
            ari_k = round(adjusted_rand_score(y, kmeans_pred), 4)
            nmi_k = round(normalized_mutual_info_score(y, kmeans_pred), 4)
        sil_k = silhouette_score(z, kmeans_pred)
        cal_k = calinski_harabasz_score(z, kmeans_pred)
        k_end = time.time()
        scores_k = {
            "kmeans_ari": ari_k,
            "kmeans_nmi": nmi_k,
            "kmeans_sil": sil_k,
            "kmeans_cal": cal_k,
            "kmeans_pred": kmeans_pred,
            "kmeans_time": k_end - k_start,
        }
        scores = {**scores, **scores_k}

    

    if plot:
        pca = PCA(2).fit_transform(z)
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.title("Ground truth")
        plt.scatter(pca[:, 0], pca[:, 1], c=y, s=4)

        plt.subplot(132)
        plt.title("K-Means pred")
        plt.scatter(pca[:, 0], pca[:, 1], c=kmeans_pred, s=4)

        plt.subplot(133)
        plt.title("Leiden pred")
        plt.scatter(pca[:, 0], pca[:, 1], c=pred, s=4)
        plt.show()
    return scores


def train(model, optim, n_epochs, dataloader, n_clusters, plot=False, save=False, cluster=["KMeans"], use_cpu=False, cluster_params ={}):
    """
    Train the graph autoencoder model (model) with the given optimizer (optim)
    for n_epochs.

    Args:
        model ([type]): [description]
        optim ([type]): [description]
        n_epochs ([type]): [description]
        dataloader ([type]): [description]
        n_clusters ([type]): [description]
        plot (bool, optional): [description]. Defaults to False.
        save (bool, optional): [description]. Defaults to False.
        cluster (list, optional): [description]. Defaults to ["KMeans"].
        use_cpu (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    device = get_device(use_cpu=use_cpu)
    losses = []
    aris_kmeans = []
    for epoch in tqdm(range(n_epochs)):
        # normalization
        model.train()
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            input_features = blocks[0].srcdata['features']
            g = blocks[-1]
            degs = g.in_degrees().float()

            adj = g.adjacency_matrix().to_dense()
            adj = adj[g.dstnodes()]
            pos_weight = torch.Tensor(
                [float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
            factor = float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
            if factor == 0:
                factor = 1
            norm = adj.shape[0] * adj.shape[0] / factor
            adj_logits, _ = model.forward(blocks, input_features)
            loss = norm * BCELoss(adj_logits, adj.to(device),
                                  pos_weight=pos_weight.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        if plot == False:
            continue
        elif epoch % plot == 0:
            score = evaluate(model, dataloader, n_clusters,
                             cluster=cluster, use_cpu=use_cpu, cluster_params = cluster_params)
            print(f'ARI {score.get("kmeans_ari")}, {score.get("kmeans_sil")}')
            aris_kmeans.append(score["kmeans_ari"])

    if plot:
        plt.figure()
        plt.plot(aris_kmeans, label="kmeans")
        plt.legend()
        plt.show()
    # return model

    score = evaluate(model, dataloader, n_clusters, save=save,
                     cluster=cluster, use_cpu=use_cpu, cluster_params = cluster_params)
    score["aris_kmeans"] = aris_kmeans
    print(f'ARI {score.get("kmeans_ari")}, {score.get("kmeans_sil")}')
    return score


def get_device(use_cpu=False):
    """[summary]

    Returns:
        [type]: [description]
    """
    if torch.cuda.is_available() and use_cpu == False:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device




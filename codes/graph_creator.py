import dgl
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap

def make_graph(X, Y=None, threshold=0, dense_dim=100,  gene_data={},
               normalize_weights="log_per_cell", nb_edges=1,
               node_features="scale", same_edge_values=False,
               edge_norm=True):
    
    num_genes = X.shape[1]

    graph = dgl.DGLGraph()
    gene_ids = torch.arange(X.shape[1], dtype=torch.int32).unsqueeze(-1)
    graph.add_nodes(num_genes, {'id': gene_ids})

    row_idx, gene_idx = np.nonzero(X > threshold)  # intra-dataset index

    if normalize_weights == "none":
        X1 = X
    if normalize_weights == "log_per_cell":
        X1 = np.log1p(X)
        X1 = X1 / (np.sum(X1, axis=1, keepdims=True) + 1e-6)

    if normalize_weights == "per_cell":
        X1 = X / (np.sum(X, axis=1, keepdims=True) + 1e-6)

    non_zeros = X1[(row_idx, gene_idx)]  # non-zero values

    cell_idx = row_idx + graph.number_of_nodes()  # cell_index
    cell_nodes = torch.tensor([-1] * len(X), dtype=torch.int32).unsqueeze(-1)

    graph.add_nodes(len(cell_nodes), {'id': cell_nodes})
    if nb_edges > 0:
        edge_ids = np.argsort(non_zeros)[::-1]
    else:
        edge_ids = np.argsort(non_zeros)
        nb_edges = abs(nb_edges)
        print(f"selecting weakest edges {int(len(edge_ids) *nb_edges)}")
    edge_ids = edge_ids[:int(len(edge_ids) * nb_edges)]
    cell_idx = cell_idx[edge_ids]
    gene_idx = gene_idx[edge_ids]
    non_zeros = non_zeros[edge_ids]

    if same_edge_values:
        graph.add_edges(
            gene_idx, cell_idx, {
                'weight':
                torch.tensor(np.ones_like(non_zeros),
                             dtype=torch.float32).unsqueeze(1)
            })
    else:
        graph.add_edges(
            gene_idx, cell_idx, {
                'weight':
                torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1)
            })

    if node_features == "scale_w_PCA":
        nX = ((X1 - np.mean(X1, axis=0))/np.std(X1, axis=0))
        gene_feat = PCA(dense_dim,  random_state=1).fit_transform(
            nX.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)
    if node_features == "scale_w_KPCA":
        nX = ((X1 - np.mean(X1, axis=0))/np.std(X1, axis=0))
        gene_feat = KernelPCA(dense_dim,  random_state=1).fit_transform(
            nX.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)
    if node_features == "scale_w_MDS":
        nX = ((X1 - np.mean(X1, axis=0))/np.std(X1, axis=0))
        gene_feat = MDS(dense_dim,  random_state=1).fit_transform(
            nX.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)
    if node_features == "scale_w_Isomap":
        nX = ((X1 - np.mean(X1, axis=0))/np.std(X1, axis=0))
        gene_feat = Isomap(n_components = dense_dim).fit_transform(
            nX.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)
    if node_features == "scale_by_cell":
        nX = ((X1 - np.mean(X1, axis=0))/np.std(X1, axis=0))
        cell_feat = PCA(dense_dim, random_state=1).fit_transform(
            nX).astype(float)
        gene_feat = X1.T.dot(cell_feat).astype(float)
    if node_features == "none":
        gene_feat = PCA(dense_dim, random_state=1).fit_transform(
            X1.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)
    if node_features == "without_pca":
        gene_feat = X1.T.astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)
        #cell_feat = X1.astype(float)


    graph.ndata['features'] = torch.cat([torch.from_numpy(gene_feat),
                                         torch.from_numpy(cell_feat)],
                                        dim=0).type(torch.float)

    graph.ndata['order'] = torch.tensor([-1] * num_genes + list(np.arange(len(X))),
                                        dtype=torch.long)  # [gene_num+train_num]
    if Y is not None:
        graph.ndata['label'] = torch.tensor([-1] * num_genes + list(np.array(Y).astype(int)),
                                            dtype=torch.long)  # [gene_num+train_num]
    else:
        graph.ndata['label'] = torch.tensor(
            [-1] * num_genes + [np.nan] * len(X))
    nb_edges = graph.num_edges()

    #if len(gene_data) != 0 and len(gene_data['gene1']) > 0:
        #graph = external_data_connections(
            #graph, gene_data, X, gene_idx, cell_idx)
    in_degrees = graph.in_degrees()
    # Edge normalization
    if edge_norm:
        for i in range(graph.number_of_nodes()):
            src, dst, in_edge_id = graph.in_edges(i, form='all')
            if src.shape[0] == 0:
                continue
            edge_w = graph.edata['weight'][in_edge_id]
            graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(
                edge_w)

    graph.add_edges(
        graph.nodes(), graph.nodes(), {
            'weight':
            torch.ones(graph.number_of_nodes(),
                       dtype=torch.float).unsqueeze(1)
        })
    return graph


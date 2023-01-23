from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import scanpy as sc

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

decode = empty_safe(np.vectorize(lambda x: x.decode("utf-8")), str)

def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data

def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = {}
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group)
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = csr_matrix(mat)
        else:
            mat = csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns

def preprocess(path, dataset_name):
    data_path = f"{path}real_data/{dataset_name}.h5"
    mat, obs, var, uns = read_data(data_path, sparsify = False, skip_exprs = False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_level = np.unique(cell_name, return_inverse= True)
    return X, cell_level

def filter_data(X, highly_genes = 500):
    X = X.astype(int)
    adata = sc.AnnData(X)
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts= 1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean = 4,  n_top_genes= highly_genes, subset = True)
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)

    return genes_idx, cells_idx

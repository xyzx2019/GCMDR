import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import matplotlib as plt
from pylab import *
import random
from inits import *

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(train_arr, test_arr):
    """Load data."""
    labels = np.loadtxt("adj.txt")    

    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(106,754)).toarray()
    logits_test = logits_test.reshape([-1,1])
#     logits_test = np.hstack((logits_test,1-logits_test))

    logits_train = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(106,754)).toarray()
    logits_train = logits_train.reshape([-1,1])
#     logits_train = np.hstack((logits_train,1-logits_train))
      
    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])

    M = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(106,754)).toarray()        
    adj = np.vstack((np.hstack((np.zeros(shape=(106,106),dtype=int),M)),np.hstack((M.transpose(),np.zeros(shape=(754,754),dtype=int)))))
    
    
    F1 = np.loadtxt("drug_feature_matrix.txt")
    F2 = np.loadtxt("ncrna_expression_full.txt")
    
    features = np.vstack((np.hstack((F1,np.zeros(shape=(106,172),dtype=int))), np.hstack((np.zeros(shape=(754,920),dtype=int), F2))))
#     features = np.ones(features.shape)
    
    features = normalize_features(features)
    size_u = F1.shape
    size_v = F2.shape

    adj = preprocess_adj(adj)
    np.savetxt('adj',adj) 
    
    return adj, features, size_u, size_v, logits_train,  logits_test, train_mask, test_mask, labels

def generate_mask(labels,N):
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(106,754)).toarray()
    mask = np.zeros(A.shape)
    while(num<5*N):
        a = random.randint(0,105)
        b = random.randint(0,753)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            num += 1
    mask = np.reshape(mask,[-1,1])
    return mask


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized


def construct_feed_dict(adj, features, labels, labels_mask, negative_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['adjacency_matrix']: adj})
    feed_dict.update({placeholders['Feature_matrix']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['negative_mask']: negative_mask})
    return feed_dict

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import torch_geometric
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import cdist
import seaborn as sns

def find_centroids(concepts, y, tau=0.5):
    concepts = concepts.detach().numpy()
    centroids = []
    used_centroid_labels = np.zeros_like(y) - 1
    centroid_labels = []

    # gets boolean encoding of concepts
    cluster_general_labels = np.unique(concepts>tau, axis=0)

    for concept in range(len(cluster_general_labels)):
        # get all concept rows where have matching boolean encoding
        cluster_samples = np.where(((concepts>tau)==(cluster_general_labels[concept])).all(axis=1))[0]

        # take mean of those activations fitting the concept
        centroid = np.mean(concepts[cluster_samples], axis=0)

        # sample - concept mapping
        used_centroid_labels[cluster_samples] = concept
        centroid_labels.append(concept)
        centroids.append(centroid)

    centroids = np.vstack(centroids)
    centroid_labels = np.stack(centroid_labels)

    return centroids, centroid_labels, used_centroid_labels

def get_top_graphs(top_indices, c, y, edges, batch):
    graphs = []
    color_maps = []
    labels = []
    concepts = []

    df = pd.DataFrame(edges)

    for idx in top_indices:
        # get neighbours
        node_indexes = torch.Tensor(list(range(batch.shape[0])))
        neighbours = node_indexes[batch == idx].numpy()
        neighbours = list(set(neighbours))

        new_G = nx.Graph()
        df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
        remaining_edges = df_neighbours.to_numpy()
        new_G.add_edges_from(remaining_edges)

        color_map = []
        color_pal = sns.color_palette("colorblind", 2).as_hex()
        for node in new_G:
            color_map.append(color_pal[1])

        color_maps.append(color_map)
        graphs.append(new_G)
        labels.append(y[idx])
        concepts.append(c[idx])


    return graphs, color_maps, labels, concepts


def get_node_distances(clustering_model, data, concepts=None, concepts2=None):
    res_sorted = cdist(data, concepts)

    return res_sorted

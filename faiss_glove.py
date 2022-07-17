import faiss
import h5py
import requests
import tempfile
import numpy as np
import os
import time


topK = 10

d = 100
M = 100
nbits = 8
index = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)

print("Downloading dataset")
with tempfile.TemporaryDirectory() as tmp:
    response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
    loc = os.path.join(tmp, "glove.hdf5")
    with open(loc, 'wb') as f:
        f.write(response.content)
    
    glove_h5py = h5py.File(loc, "r")
    
list(glove_h5py.keys())

dataset = np.array(glove_h5py['train'])
queries = np.array(glove_h5py['test'])
print(dataset.shape)
print(queries.shape)

normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]


def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

print("training...")
index.train(dataset)
#index.train(normalized_dataset)

print("adding...")
index.add(dataset)
#index.add(normalized_dataset)

print("searching")
D, I = index.search(queries, topK)
#I = index.search(queries, topK, faiss.METRIC_INNER_PRODUCT)
print(I[:10], D[:10])
print(glove_h5py['neighbors'][:10,:topK])

print("Recall:", compute_recall(I, glove_h5py['neighbors'][:, :topK]))

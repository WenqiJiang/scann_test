import numpy as np
import h5py
import os
import requests
import tempfile
import time

import scann

topK = 10

with tempfile.TemporaryDirectory() as tmp:
    response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
    loc = os.path.join(tmp, "glove.hdf5")
    with open(loc, 'wb') as f:
        f.write(response.content)
    
    glove_h5py = h5py.File(loc, "r")
    
list(glove_h5py.keys())

dataset = glove_h5py['train']
queries = glove_h5py['test']
print(dataset.shape)
print(queries.shape)

normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README

# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
searcher = scann.scann_ops_pybind.builder(normalized_dataset, topK, "dot_product").score_ah(
    1, anisotropic_quantization_threshold=0.0).build()
#searcher = scann.scann_ops_pybind.builder(normalized_dataset, topK, "dot_product").score_ah(
#    2, anisotropic_quantization_threshold=0.2).build()

def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

print("Query shape: {}".format(queries.shape))
qnum, dim = queries.shape
print("Batched Search Parallel: ")
# this will search the top 100 of the 2000 leaves, and compute
# the exact dot products of the top 100 candidates from asymmetric
# hashing to get the final top 10 candidates.
start = time.time()
neighbors, distances = searcher.search_batched_parallel(queries)
end = time.time()

# we are given top 100 neighbors in the ground truth, so select top 10
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :topK]))
print("Time:", end - start)
print("QPS: ", qnum / (end - start))

print("Batched Search: ")
# this will search the top 100 of the 2000 leaves, and compute
# the exact dot products of the top 100 candidates from asymmetric
# hashing to get the final top 10 candidates.
start = time.time()
neighbors, distances = searcher.search_batched(queries)
end = time.time()

# we are given top 100 neighbors in the ground truth, so select top 10
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :topK]))
print("Time:", end - start)
print("QPS: ", qnum / (end - start))

print("Single-query Search: ")
# we have been exclusively calling batch search so far; the single-query call has the same API
start = time.time()
neighbors, distances = searcher.search(queries[0], final_num_neighbors=topK)
end = time.time()

#print("Time:", end - start)
print("Latency (ms):", 1000*(end - start))
print("QPS: ", 1 / (end - start))

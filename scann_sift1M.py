import numpy as np
import os
import time
import sys

import scann

topK = 100


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Wenqi: Format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

print("Loading sift1M...", end='', file=sys.stderr)
xt = fvecs_read("sift/sift_learn.fvecs")
xb = fvecs_read("sift/sift_base.fvecs")
xq = fvecs_read("sift/sift_query.fvecs")
gt = ivecs_read("sift/sift_groundtruth.ivecs")
print("done", file=sys.stderr)

print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape, gt.shape))

nq, d = xq.shape
nb, d = xb.shape
assert gt.shape[0] == nq

# normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README

# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
searcher = scann.scann_ops_pybind.builder(xb, topK, "squared_l2").score_ah(
    8, anisotropic_quantization_threshold=0.0).build()
#searcher = scann.scann_ops_pybind.builder(normalized_dataset, topK, "dot_product").score_ah(
#    2, anisotropic_quantization_threshold=0.2).build()

def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

print("Batched Search Parallel: ")
# this will search the top 100 of the 2000 leaves, and compute
# the exact dot products of the top 100 candidates from asymmetric
# hashing to get the final top 10 candidates.
start = time.time()
neighbors, distances = searcher.search_batched_parallel(xq)
end = time.time()

# we are given top 100 neighbors in the ground truth, so select top 10
print("Time:", end - start)
print("QPS: ", nq / (end - start))


if topK >= 1:
    print("R@1:", compute_recall(neighbors[:,:1], gt[:, :1]))
if topK >=10:
    print("R@10:", compute_recall(neighbors[:,:10], gt[:, :10]))
if topK >=100:
    print("R@100:", compute_recall(neighbors[:,:100], gt[:, :100]))

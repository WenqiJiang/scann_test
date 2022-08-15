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


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

print("Loading sift1B...", end='', file=sys.stderr) 
xb = mmap_bvecs('/data/bigann/bigann_base.bvecs')
xq = mmap_bvecs('/data/bigann/bigann_query.bvecs')
xt = mmap_bvecs('/data/bigann/bigann_learn.bvecs')
gt = ivecs_read('/data/bigann/gnd/idx_1000M.ivecs')
print("done", file=sys.stderr)

print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape, gt.shape))

nq, d = xq.shape
nb, d = xb.shape
assert gt.shape[0] == nq
assert nb = int(1e9)

# normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README

nlist = 32768 # number of clusters
nprobe = 32
N_bytes = 32 # bytes per quantized vector
assert d % (2 * N_bytes) == 0
dim_per_4bit_PQ = int(d / (2 * N_bytes))

index_parent_dir = './scann_indexes'
index_name = 'SIFT1B_IVF{},PQ{}'.format(nlist, N_bytes)
index_path = os.path.join(index_parent_dir, index_name)

training_sample_size = n_train = max(256 * 1000, 100 * nlist) # same as Faiss

if not os.path.exists(index_path):
    os.mkdir(index_path)
    # use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
    searcher = scann.scann_ops_pybind.builder(xb, topK, "squared_l2").tree(
        num_leaves=nlist, num_leaves_to_search=nlist, training_sample_size=training_sample_size).score_ah(
        dim_per_4bit_PQ, anisotropic_quantization_threshold=float("nan"), training_sample_size=training_sample_size).build()
    # save index
    searcher.serialize(index_path)
else:
    searcher = scann.scann_ops_pybind.load_searcher(index_path)

def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

def compute_recall_at_1(neighbors, true_neighbors):
    total = 0
    correct_cnt = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += 1
        correct_cnt += np.intersect1d(np.array(gt_row[0]), row).shape[0]
    return correct_cnt / total

print("Batched Search Parallel: ")
# this will search the top 100 of the 2000 leaves, and compute
# the exact dot products of the top 100 candidates from asymmetric
# hashing to get the final top 10 candidates.
start = time.time()
neighbors, distances = searcher.search_batched_parallel(xq, leaves_to_search=nprobe)
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
if topK >= 1:
    print("R1@1:", compute_recall_at_1(neighbors[:,:1], gt[:, :1]))
if topK >=10:
    print("R1@10:", compute_recall_at_1(neighbors[:,:10], gt[:, :10]))
if topK >=100:
    print("R1@100:", compute_recall_at_1(neighbors[:,:100], gt[:, :100]))

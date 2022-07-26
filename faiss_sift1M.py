from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
from multiprocessing.dummy import Pool as ThreadPool


topK          = 100
dbname        = "SIFT1M"
nbytes        = 16
nbits         = 4
assert nbytes * 8 % nbits == 0
m             = int(nbytes * 8 / nbits)
index_key     = "PQ{}bytes,{}bits".format(nbytes, nbits)

tmpdir = './trained_CPU_indexes/bench_cpu_{}_{}'.format(dbname, index_key)

### Wenqi: when loading the index, save it to numpy array, default: False
save_numpy_index = False
# save_numpy_index = False 
# we mem-map the biggest files to avoid having them in memory all at
# once


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

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

#################################################################
# Bookkeeping
#################################################################


if not os.path.isdir(tmpdir):
    print("%s does not exist, creating it" % tmpdir)
    os.mkdir(tmpdir)


#################################################################
# Prepare dataset
#################################################################


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


#################################################################
# Training
#################################################################


def get_trained_index():
    filename = "%s/%s_%s_trained.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = faiss.IndexPQ(d, m, nbits)

        # make sure the data is actually in RAM and in float
        index.verbose = True

        t0 = time.time()
        index.train(xt)
        index.verbose = False
        print("train done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


#################################################################
# Adding vectors to dataset
#################################################################

def rate_limited_imap(f, l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


def matrix_slice_iterator(x, bs):
    " iterate over the lines of x in blocks of size bs"
    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    return rate_limited_imap(
        lambda i01: x[i01[0]:i01[1]].astype('float32').copy(),
        block_ranges)


def get_populated_index():

    filename = "%s/%s_%s_populated.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = get_trained_index()
        i0 = 0
        t0 = time.time()
        for xs in matrix_slice_iterator(xb, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            index.add(xs)
            i0 = i1
        print()
        print("Add done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
        if save_numpy_index:
            print("Saving index to numpy array...")
            chunk = faiss.serialize_index(index)
            np.save("{}.npy".format(filename), chunk)
            print("Finish saving numpy index")
    return index

def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

#################################################################
# Perform searches
#################################################################

index = get_populated_index()

# make sure queries are in RAM
xq = xq.astype('float32').copy()

# we do queries in a single thread
#faiss.omp_set_num_threads(1)

print("Searching")
sys.stdout.flush()
t0 = time.time()
D, I = index.search(xq, 100)
t1 = time.time()
print("time = %8.3f  sec" % ((t1 - t0)))
print("QPS = %8.2f  " % (nq/(t1 - t0)))

if topK >= 1:
    print("R@1:", compute_recall(I[:,:1], gt[:, :1]))
if topK >=10:
    print("R@10:", compute_recall(I[:,:10], gt[:, :10]))
if topK >=100:
    print("R@100:", compute_recall(I[:,:100], gt[:, :100]))

gpu = True
try:
    import cupy as xp
except:
    import numpy as xp
    gpu = False

import gc

def make_dots(arglist):
    # multiprocessing.pool.map() only works with one argument
    points, d, num_p = arglist
    points = int(points / num_p)
    # generate all points
    # # on a system with 16 GB RAM and 12 CPUs / workers,
    # # system memory is exhausted when the pts array
    # # is over 500 MB, or 60 million elements (points * dims).
    # # Half of RAM is taken by pts arrays in all workers.
    # # TODO: implement code to fragment this array when too big
    pts = xp.random.random_sample((points, d)) - 0.5
    # keep a sample of points
    ssize = int(100 / num_p)
    pts_sample = pts[:ssize, :]
    # calculate distances from center
    pts = xp.power(pts, 2)
    dists = xp.sum(pts, axis = 1)
    del pts
    gc.collect()
    ## square root of sum - it's the distance to origin
    dists = xp.power(dists, 0.5)
    # this many are within the sphere (distance to origin  <= 0.5)
    p_int = (dists <= 0.5).sum()
    del dists
    gc.collect()
    if gpu:
        return [xp.asnumpy(p_int), xp.asnumpy(pts_sample)]
    else:
        return [p_int, pts_sample]

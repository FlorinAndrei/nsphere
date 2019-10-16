import gc
import random

def make_dots(arglist):
    # multiprocessing.pool.map() only works with one argument
    points, d, num_p, sysmem, gpumem, pointloops = arglist
    if gpumem > 0:
        import cupy as xp
        usemem = gpumem
    else:
        import numpy as xp
        usemem = sysmem / 2
        # Numpy bug: on Mac/Linux with multiprocessing
        # all workers are seeded the same random state, so the "random"
        # sequences made by different workers are identical. WTH
        # Have to seed each worker from the Python random lib.
        rseed = random.randint(0, 4294967296)
        xp.random.seed(rseed)
    points = int(points / num_p)
    # generate all points
    # # on a system with 16 GB RAM and 12 CPUs / workers,
    # # system memory is exhausted when the pts array
    # # is over 500 MB, or 60 million elements (points * dims).
    # # Half of RAM is taken by pts arrays in all workers.
    # # TODO: implement code to fragment this array when too big

    # 0.04 is a magic number based on
    # the size of each cell in the pts matrix (float32 usually)
    magic_ratio = 0.04
    if (points * d / usemem) > magic_ratio:
        split = True

    p_int = 0
    for i in range(pointloops):
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
        p_int += (dists <= 0.5).sum()
        del dists
        gc.collect()
    p_int = p_int / pointloops
    if gpumem > 0:
        return [xp.asnumpy(p_int), xp.asnumpy(pts_sample)]
    else:
        return [p_int, pts_sample]

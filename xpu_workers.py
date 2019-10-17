import gc
import random

def make_dots(arglist):
    # multiprocessing.pool.map() only works with one argument
    # so we pass a tuple of arguments as "one argument"
    points, d, workers, sysmem, gpumem = arglist

    if gpumem > 0:
        import cupy as xp
        # use all the GPU memory
        usemem = gpumem
    else:
        import numpy as xp
        # on a laptop don't use all the system memory
        usemem = int(sysmem * 0.5)
        # If you use multiprocessing.set_start_method('spawn') (the default on Windows)
        # then the following lines are not necessary. If you use the 'fork' method instead
        # (default on Unix) then all workers will generate identical "random" sequences,
        # lowering the entropy of the population - then you must seed each worker randomly
        # as shown here:
        #rseed = random.randint(0, 4294967296)
        #xp.random.seed(rseed)

    # if workers > 1, we have more than one worker
    # tell the worker to divide its share of calculations
    points = int(points / workers)

    # this is a magic number based on
    # the size of each cell in the pts matrix (float32 usually)
    # TODO: compute magic_ratio from array type
    #       calculate array size, then use a percentage instead of magic_ratio
    magic_ratio = 0.03

    # too many points and too many dimensions - it won't fit in memory
    # so then split the job in multiple worker_loops
    # each loop ought to fit in memory (see magic_ratio)
    data_vs_mem = points * d * workers / usemem
    if data_vs_mem > magic_ratio:
        split = True
        worker_loops = int(data_vs_mem / magic_ratio) + 1
        points_per_loop = int(points / worker_loops)
        final_loop_points = points - points_per_loop * (worker_loops - 1)
        points = points_per_loop
    else:
        split = False
        worker_loops = 1
        points_per_loop = points
        final_loop_points = 0

    # initialize number of points inside the sphere
    p_int = 0

    for i in range(worker_loops):
        if split and i == worker_loops - 1 and final_loop_points != 0:
            # this is the last loop and the remainder is not zero
            # TODO: the remainder should never be zero - seems like logic bug
            pts = xp.random.random_sample((final_loop_points, d)) - 0.5
        else:
            pts = xp.random.random_sample((points_per_loop, d)) - 0.5
        if i == 0:
            # set aside a sample of points for display
            # total sample size from all workers ~= 100 points
            ssize = int(100 / workers)
            pts_sample = pts[:ssize, :]

        # calculate distances-squared from center
        pts = xp.power(pts, 2)
        dists = xp.sum(pts, axis = 1)
        del pts
        gc.collect()

        ## square root of sum - it's the distance to origin
        dists = xp.power(dists, 0.5)
        # this many points are within the sphere (distance to origin  <= 0.5)
        p_int_sum = (dists <= 0.5).sum()
        p_int = p_int + p_int_sum
        del dists
        gc.collect()

    #p_int = p_int / worker_loops
    if gpumem > 0:
        return [p_int, xp.asnumpy(pts_sample)]
    else:
        return [p_int, pts_sample]

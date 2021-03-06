# nsphere

In 2 dimensions (N = 2), consider the unit square (1 x 1 size). In it, inscribe a circle. What is the ratio between the area of the circle and the area of the square?

![empty slice](/images/empty_slice.png)

In 3 dimensions (N = 3), consider the unit cube (1 x 1 x 1). In it, inscribe a sphere. What is the ratio between the volume of the sphere and the volume of the cube?

What happens when N = 4? N = 5? Or when N is arbitrarily large?

Does the volume of the inscribed N-sphere, compared to the volume of the N-cube, change when N varies? Is there a trend?

Would the volume ratios tend to a fixed value? If so, what is that value? Is it zero, or some other number?

## Monte Carlo

A good approximation of the volume ratio for any value of N can be obtained with the Monte Carlo method.

Generate a large number of dots within the cube, with random locations. Let's say the total number of dots is D.

Out of these dots, some will be inside the sphere, others will be outside. Count all the dots that are inside the sphere - let's say their total number is Ds.

![slice with dots](/images/slice_with_dots.png)

If you generate a very large number of dots, and they are random enough, then the sphere / cube volume ratio is approximated by:

```
ratio = Ds / D
```

It's an "experimental" method that does not rely on exact analytic solutions.

## Results

It turns out, as N increases, the volume ratio decreases. As the number of dimensions keeps getting larger, the sphere appears to "shrink" in volume, as compared to the cube, even though the sphere is always inscribed in the cube.

![graph with ratios](/images/graph_with_ratios.png)

The decrease is sharp. Beyond N = 10 the volume ratio is essentially zero.

Seems like, as N increases, there is more space available in the corners of the cube. The volume outside the sphere, in the corners of the cube containing it, becomes larger and larger. In comparison, the sphere becomes more and more insignificant.

Makes sense when you think about it for a while, but it's quite surprising at first.

## Code

Check the [nsphere.ipynb](https://github.com/FlorinAndrei/nsphere/blob/master/nsphere.ipynb) Jupyter notebook in this repository. It contains the Monte Carlo simulation and the visualization code that were used to create the images in this document. You could download and run it in your own Jupyter installation. Or run it on nbviewer (which is the more reliable way to read the source code of the notebook, compared to GitHub's own notebook viewer):

[https://nbviewer.jupyter.org/github/FlorinAndrei/nsphere/blob/master/nsphere.ipynb](https://nbviewer.jupyter.org/github/FlorinAndrei/nsphere/blob/master/nsphere.ipynb)

The Numpy / CUDA implementation is in [xpu_workers.py](https://github.com/FlorinAndrei/nsphere/blob/master/xpu_workers.py).

### Vector operations

The current version of the notebook uses vector operations via Numpy. This is **much** faster than iterative loops. It allows raising the precision (using more dots) without the execution time becoming too huge.

It does use a lot of memory, however. This is why we're forcing garbage collection in a bunch of places. This ought to keep memory usage within the limits we're enforcing (see below).

### Multiprocessing

To speed things up even further, the main Numpy code that does the simulation is instantiated by multiple workers at once, one for each CPU. This provides another massive speed boost.

### GPU

If Cupy is installed and a GPU is available, the worker (only 1 worker is used when GPU is enabled) will switch from Numpy to Cupy and will use the GPU for linear algebra. Once the worker is done, the matrices with the results are returned to the host in plain Numpy form.

The code assumes either 0 or 1 GPUs are installed on the system. It cannot use multiple GPUs for now.

### Limits

After all optimizations, the main bottleneck is memory. All math is done in Numpy / Cupy arrays that use most of the available system / GPU RAM, vectorized and parallel, for the greatest speed possible. The code auto-detects the system and GPU memory sizes, and it will adapt to the existing memory: if the whole matrix with all the points does not fit in memory, it will fragment it and loop until all fragments are processed.

This way simulations could run using a very large number of points (a billion or more) - the code would just have to loop until all points are processed. Time is traded for precision.

On a GPU, the app will use most of its memory. On the CPU, it will use somewhat less than half the system RAM.

# nsphere

In 2 dimensions (N = 2), consider the unity square (1 x 1 size). In it, inscribe a circle. What is the ratio between the area of the circle and the area of the square?

![empty slice](/images/empty_slice.png)

In 3 dimensions (N = 3), consider the unity cube (1 x 1 x 1). In it, inscribe a sphere. What is the ratio between the volume of the sphere and the volume of the cube?

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

The decrease is sharp. Around N = 16, the volume ratio is essentially zero already.

Seems like, as N increases, there is more space available in the corners of the cube. The volume outside the sphere, in the corners of the cube containing it, becomes larger and larger. In comparison, the sphere becomes more and more insignificant.

Makes sense when you think about it for a while, but it's quite surprising at first.

## Code

Check the [nsphere.ipynb](https://github.com/FlorinAndrei/nsphere/blob/master/nsphere.ipynb) Jupyter notebook in this repository. It contains the Monte Carlo simulation and the visualization code that were used to create the images in this document. You could download and run it in your own Jupyter installation. Or run it on nbviewer:

[https://nbviewer.jupyter.org/github/FlorinAndrei/nsphere/blob/master/nsphere.ipynb](https://nbviewer.jupyter.org/github/FlorinAndrei/nsphere/blob/master/nsphere.ipynb)

### Vector operations

The current version of the notebook uses vector operations via Numpy. This is **much** faster than iterative loops. It allows raising the precision (using more dots) without the execution time becoming too huge.

import numpy as np

def dist(x, c):
    """ Calculates the pairwise Euclidian distance between every row in x and every row in c.

    Inputs:
    - x: NxD matrix where each row represents a feature vector
    - c: MxD matrix where each row represents a feature vector

    Outputs:
    - d: NxM where the value at (i, j) represents the Euclidian distance between features x_i and c_j
    """
    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if dimx != dimc:
        raise NameError("Data dimension does not match dimension of centres")

    n2 = (
        np.transpose(
            np.dot(
                np.ones((ncentres, 1)),
                np.transpose(np.sum(np.square(x), 1).reshape(ndata, 1)),
            )
        )
        + np.dot(
            np.ones((ndata, 1)),
            np.transpose(np.sum(np.square(c), 1).reshape(ncentres, 1)),
        )
        - 2 * np.dot(x, np.transpose(c))
    )

    n2[n2 < 0] = 0
    return n2

def new_point(old, new, **kwargs):
    dists = dist(new, old)

    new = ~np.sum(np.isclose(dists, np.zeros_like(dists),**kwargs), axis=1, dtype=bool)

    return new
import numpy as np
from params import nx

vj = 2*np.pi*(np.arange(nx + 1))
vj[0] = np.pi/4


def setrho(x, qe):
    rhoj = np.zeros_like(vj)

    valid = ~np.isnan(x)
    x = x[valid]

    ij = np.floor(x).astype(int)
    denom = 2*ij + 1

    wR = (x**2 - ij**2) / denom
    wL = 1.0 - wR

    np.add.at(rhoj, ij, qe * wL)
    np.add.at(rhoj, ij + 1, qe * wR)

    rhoj /= vj

    return rhoj

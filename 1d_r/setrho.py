import numpy as np
from params import nx

vj = 2*np.pi*(np.arange(nx + 1))
vj[0] = np.pi/4


def setrho(x, qe):
    rhoj = np.zeros_like(vj)

    valid = ~np.isnan(x)
    x = x[valid]

    ij = np.floor(x).astype(int)
    ij1 = ij + 1
    area = ij1**2 - ij**2
    wL = (x**2-ij**2)/area
    wR = 1.0 - wL

    np.add.at(rhoj, ij, qe*wR)
    np.add.at(rhoj, ij1, qe*wL)

    rhoj /= vj

    return rhoj

import numpy as np
from params import nx, x0
from setrho import vj


def curnt(x, v, q):

    curnt = np.zeros(nx + 1)
    ij = np.floor(x).astype(np.int64)
    ij1 = ij + 1
    area = 2*ij + 1
    delta = x - ij
    wL = delta*(delta + 2*ij)/area
    wR = 1.0 - wL

    np.add.at(curnt, ij-x0, q*wR*v)
    np.add.at(curnt, ij1-x0, q*wL*v)

    curnt /= vj

    return curnt

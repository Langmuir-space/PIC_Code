import numpy as np
from params import nx, qdx


def setrho(x):
    rho0 = np.zeros(nx)

    ij = np.floor(x).astype(int)
    delx = x - ij
    ij1 = (ij + 1) % nx

    np.add.at(rho0, ij, (1 - delx)*qdx)
    np.add.at(rho0, ij1, delx*qdx)

    return rho0

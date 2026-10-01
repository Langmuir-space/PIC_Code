import numpy as np
from params import nx, x0, dx

index = np.arange(x0, x0+nx+1, 1)
vj = 2*np.pi*index*dx**2
if index[0] == 0:
    vj[0] = 0.25*np.pi*dx**2


def setrho(x, q):
    rhoj = np.zeros(nx + 1)
    ij = np.floor(x).astype(int)
    ij1 = ij + 1
    area = 2*ij + 1
    wL = (x**2-ij**2)/area
    wR = 1.0 - wL

    np.add.at(rhoj, ij-x0, q*wR)
    np.add.at(rhoj, ij1-x0, q*wL)

    rhoj /= vj
    return rhoj

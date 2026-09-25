import numpy as np
from params import nx, x0

index = np.arange(x0, x0+nx+1, 1)
vj = 2*np.pi*index
if index[0] == 0:
    vj[0] = np.pi/4

def curnt(x, vy, vz, q):

    jy = np.zeros(nx + 1)
    jz = np.zeros(nx + 1)
    ij = np.floor(x).astype(int)
    ij1 = ij + 1
    area = 2*ij + 1
    wL = (x**2-ij**2)/area
    wR = 1.0 - wL

    np.add.at(jy, ij-x0, q*wR*vy)
    np.add.at(jy, ij1-x0, q*wL*vy)

    np.add.at(jz, ij-x0, q*wR*vz)
    np.add.at(jz, ij1-x0, q*wL*vz)

    jy /= vj
    jz /= vj

    return jy, jz

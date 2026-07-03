import numpy as np
from params import nx, xmax, dx


def curnt(x, vx, vy, vz, qdx):
    rho = np.zeros(nx)
    jym = np.zeros(nx)
    jzm = np.zeros(nx)
    jyp = np.zeros(nx)
    jzp = np.zeros(nx)

    ij = np.floor(x).astype(int)
    delx = x - ij
    ij1 = (ij + 1) % nx

    np.add.at(jym, ij, (1 - delx)*vy*qdx)
    np.add.at(jym, ij1, delx*vy*qdx)

    np.add.at(jzm, ij, (1 - delx)*vz*qdx)
    np.add.at(jzm, ij1, delx*vz*qdx)

    x += vx     # dt = dx   x(n)Δt → x(n + 1)Δt
    x[x > nx] -= nx
    x[x < 0] += nx

    ij = np.floor(x).astype(int)
    delx = x - ij
    ij1 = (ij + 1) % nx

    np.add.at(rho, ij, (1 - delx)*qdx)
    np.add.at(rho, ij1, delx*qdx)

    np.add.at(jyp, ij, (1 - delx)*vy*qdx)
    np.add.at(jyp, ij1, delx*vy*qdx)

    np.add.at(jzp, ij, (1 - delx)*vz*qdx)
    np.add.at(jzp, ij1, delx*vz*qdx)

    return jym, jzm, jyp, jzp, rho, x

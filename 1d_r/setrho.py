import numpy as np
from params import nx, x0, dx

index = np.arange(x0, x0+nx+1, 1)
vj = 2*np.pi*index*dx**2
if index[0] == 0:
    vj[0] = 0.25*np.pi*dx**2


def setrho(x, q):

    rhoj = np.zeros(nx + 1)
    ij = np.floor(x).astype(np.int64)
    ij1 = ij + 1
    area = 2*ij + 1
    delta = x - ij
    wL = delta*(delta + 2*ij)/area
    wR = 1.0 - wL

    # print("x dtype:", x.dtype)
    # print("ij dtype:", ij.dtype)
    # print("x type:", type(x[0]))
    # print("ij type:", type(ij[0]))

    # print("x squared:", x[:5]**2)
    # print("ij :", ij[:5])
    # print("ij squared:", ij[:5]**2)
    # print("difference:", x[:5]**2 - ij[:5]**2)

    # delta = x - ij

    # num1 = x**2 - ij**2
    # num2 = delta * (2 * ij + delta)

    # print("x:", x[:5])
    # print("ij:", ij[:5])
    # print("delta:", delta[:5])
    # print("num1:", num1[:5])
    # print("num2:", num2[:5])
    # print("wL:", (num1 / (2 * ij + 1))[:5])
    # print("wL stable:", (num2 / (2 * ij + 1))[:5])

    np.add.at(rhoj, ij-x0, q*wR)
    np.add.at(rhoj, ij1-x0, q*wL)
    # print('x**2', (x**2).min(), (x**2).max())
    # print("x**2-ij**2 min/max:", (x**2-ij**2).min(), (x**2-ij**2).max())
    # print("wL min/max:", wL.min(), wL.max())
    # print("wR min/max:", wR.min(), wR.max())
    # print("area min/max:", area.min(), area.max())

    # print("x range:", x.min(), x.max())
    # print("q range:", np.min(q), np.max(q))
    # print("deposited charge sum:", rhoj.sum())
    # print("vj range:", vj.min(), vj.max())
    # print("rho before normalization:", rhoj.min(), rhoj.max())

    rhoj /= vj

    # print("rho after normalization:", rhoj.min(), rhoj.max())
    # print("max |rho|:", np.max(np.abs(rhoj)))
    return rhoj

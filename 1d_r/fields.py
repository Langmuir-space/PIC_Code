import numpy as np
from params import nx, dx, dt, dtdr
from utils import tdma_pre, tdma_solve
from setrho import index

a = np.zeros(nx-1)
b = np.zeros(nx-1)
c = np.zeros(nx-1)
a[0] = 0
a[1:] = - (1 - 1/(2*index[2:-1]))
b[:] = 2
c[:-1] = - (1 + 1/(2*index[1:-2]))
c[-1] = 0
bp, cp = tdma_pre(a, b, c)


def field_energy(ex, ey, ez, by, bz):
    return np.sum(ex**2), np.sum(ey**2), np.sum(ez**2), \
        np.sum(by**2), np.sum(bz**2)


def field(jy, jz, rho):
    rhog = rho*(dx**2)
    phi = np.zeros(nx + 1)
    phi[0] = 0
    phi[1:-1] = tdma_solve(a, bp, cp, rhog[1:-1])
    phi[-1] = 0
    ex_half = np.zeros(nx)
    ex_half[:] = - (phi[1:] - phi[:-1])/dx
    ex = np.zeros(nx + 1)
    ex[0] = 0
    ex[1:-1] = (1 + 1/(2*index[1:-1]))*ex_half[1:]/2 \
        + (1 - 1/(2*index[1:-1]))*ex_half[:-1]/2
    ex[-1] = 0

    return ex, phi


def ftdt(ey, ez, byh, bzh, jy, jz):
    byh += 0.5*dtdr*(ez[1:] - ez[:-1])
    bzh += - 0.5*dtdr*(index[1:]*ey[1:] - index[:-1]*ey[:-1])/(index[:-1] + 0.5)
    ey[1:-1] += - dtdr*(bzh[1:] - bzh[:-1]) - dt*jy[1:-1]
    ez[1:-1] += dtdr*((index[1:-1] + 0.5)*byh[1:] - (index[1:-1] - 0.5)*byh[:-1])/index[1:-1] - dt*jz[1:-1]
    ey[0] = 0; ey[-1] = 0
    ez[0] = 0; ez[-1] = 0
    byh += 0.5*dtdr*(ez[1:] - ez[:-1])
    bzh += - 0.5*dtdr*(index[1:]*ey[1:] - index[:-1]*ey[:-1])/(index[:-1] + 0.5)
    return ey, ez, byh, bzh


def convert(byh, bzh):
    by = np.zeros(nx + 1)
    bz = np.zeros(nx + 1)
    by[1:-1] = ((index[1:-1] + 0.5)*byh[1:] + (index[1:-1] - 0.5)*byh[:-1])/(2*index[1:-1])
    bz[1:-1] = (bzh[1:] + bzh[:-1])/2
    by[0] = by[1]; by[-1] = by[-2]
    bz[0] = bz[1]; bz[-1] = bz[-2]
    return by, bz


def field_ex(rho):
    rhog = rho*(dx**2)
    phi = np.zeros(nx + 1)
    phi[0] = 0
    phi[1:-1] = tdma_solve(a, bp, cp, rhog[1:-1])
    phi[-1] = 0
    ex_half = np.zeros(nx)
    ex_half[:] = - (phi[1:] - phi[:-1])/dx
    ex = np.zeros(nx + 1)
    ex[0] = 0
    ex[1:-1] = (1 + 1/(2*index[1:-1]))*ex_half[1:]/2 + (1 - 1/(2*index[1:-1]))*ex_half[:-1]/2
    ex[-1] = 0
    return ex, phi

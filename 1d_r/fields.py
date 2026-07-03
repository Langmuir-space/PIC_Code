import numpy as np
from params import nx, dx, dt
from utils import tdma_pre, tdma_solve

k = 2*np.pi*np.fft.fftfreq(nx, d=dx)
aa = np.zeros(nx)
bb = np.sin(k*dx)/dx
aa[1:nx] = (dx/2/np.sin(k[1:nx]*dx/2))**2
tem = 0.25*dt

ij = np.arange(nx)
a = np.zeros(nx)
b = np.zeros(nx)
c = np.zeros(nx)
a[0] = 0
b[0] = 4
c[0] = -4
a[1:] = - (1 - 1/(2*ij[1:]))
b[1:] = 2
c[1:] = - (1 + 1/(2*ij[1:]))

bp, cp = tdma_pre(a, b, c)


def field_energy(ex, ey, ez, by, bz):
    return np.sum(ex**2), np.sum(ey**2), np.sum(ez**2), \
        np.sum(by**2), np.sum(bz**2)


def field(jym, jzm, jyp, jzp, rho, eyl, eyr, ezl, ezr):
    rhok = np.fft.fft(rho)
    phik = aa*rhok
    exk = -1j*bb*phik
    ex = np.real(np.fft.ifft(exk))

    eyl -= tem*jym
    eyr -= tem*jym
    ezl -= tem*jzm
    ezr -= tem*jzm
    eyl = np.roll(eyl, -1)
    eyr = np.roll(eyr, 1)
    ezl = np.roll(ezl, -1)
    ezr = np.roll(ezr, 1)
    eyl -= tem*jyp
    eyr -= tem*jyp
    ezl -= tem*jzp
    ezr -= tem*jzp

    ey = eyr + eyl
    bz = eyr - eyl
    ez = ezr + ezl
    by = ezl - ezr

    return ex, ey, ez, by, bz, eyl, eyr, ezl, ezr


def field_ex(rho):
    phi = np.zeros(nx + 1)
    phi[:-1] = tdma_solve(a, bp, cp, rho[:-1])
    phi[-1] = 0     # Boundary Condition
    ex_half = np.zeros(nx)
    ex_half[:] = phi[1:] - phi[:-1]
    ex = np.zeros(nx + 1)
    ex[0] = ex_half[0]
    ex[1:-1] = (1 + 1/ij[1:])*ex_half[1:]/2 + (1 - 1/ij[1:])*ex_half[:-1]/2
    ex[-1] = 0
    return ex, phi

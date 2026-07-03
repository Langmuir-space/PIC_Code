import numpy as np
from params import nx, dx, dt

k = 2*np.pi*np.fft.fftfreq(nx, d=dx)
aa = np.zeros(nx)
bb = np.sin(k*dx)/dx
aa[1:nx] = (dx/2/np.sin(k[1:nx]*dx/2))**2
tem = 0.25*dt


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

    # return ex, ey, ez, by, bz, eyl, eyr, ezl, ezr
    return ex  # for electrostatic

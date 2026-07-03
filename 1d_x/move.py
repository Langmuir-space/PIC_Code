import numpy as np
from params import nx


def move(vx0, vy0, vz0, gamma0, ae, tx, tz, x,
         ex, ey, ez, by, bz):

    aex = ae*ex
    aey = ae*ey
    aez = ae*ez
    aby = ae*by
    abz = ae*bz
    ake = 0

    ij = np.floor(x).astype(int)
    delx = x - ij
    ij1 = (ij + 1) % nx

    aexpt = aex[ij] + delx*(aex[ij1] - aex[ij])
    aeypt = aey[ij] + delx*(aey[ij1] - aey[ij])
    aezpt = aez[ij] + delx*(aez[ij1] - aez[ij])
    abypt = aby[ij] + delx*(aby[ij1] - aby[ij])
#     abzpt = abz[ij] + delx*(abz[ij1] - abz[ij]) + tz
    abzpt = abz[ij] + delx*(abz[ij1] - abz[ij])

    gvxs = vx0*gamma0 + aexpt
    gvys = vy0*gamma0 + aeypt
    gvzs = vz0*gamma0 + aezpt

    gamma = np.sqrt(1. + gvxs*gvxs + gvys*gvys + gvzs*gvzs)
    ake = np.sum(gamma - 1)
    dcg = 1. / gamma
    abxpt = dcg*tx
    abypt = dcg*abypt
    abzpt = dcg*abzpt

    vvx = gvxs + gvys*abzpt - gvzs*abypt
    vvy = gvys + gvzs*abxpt - gvxs*abzpt
    vvz = gvzs + gvxs*abypt - gvys*abxpt

    f = 2./(1. + abxpt*abxpt + abypt*abypt + abzpt*abzpt)
    abxpt = f*abxpt
    abypt = f*abypt
    abzpt = f*abzpt
    gvxs = gvxs + vvy*abzpt - vvz*abypt + aexpt
    gvys = gvys + vvz*abxpt - vvx*abzpt + aeypt
    gvzs = gvzs + vvx*abypt - vvy*abxpt + aezpt

    gamma = np.sqrt(1. + gvxs*gvxs + gvys*gvys + gvzs*gvzs)
    vx = gvxs / gamma
    vy = gvys / gamma
    vz = gvzs / gamma

    return vx, vy, vz, gamma, ake

import numpy as np


def move(vx0, vy0, vz0, gamma0, ae, tx, tz, x,
         ex, ey, ez, by, bz):

    x_original = x
    aex = ae*ex
    aey = ae*ey
    aez = ae*ez
    aby = ae*by
    abz = ae*bz
    ake = 0

    valid = ~np.isnan(x)
    x = x[valid]
    vx0 = vx0[valid]
    vy0 = vy0[valid]
    vz0 = vz0[valid]
    gamma0 = gamma0[valid]

    ij = np.floor(x).astype(int)
    ij1 = ij + 1
    area = ij1**2 - ij**2
    wL = (x**2 - ij**2)/area
    wR = 1.0 - wL

    aexpt = wR*aex[ij] + wL*aex[ij1]
    aeypt = wR*aey[ij] + wL*aey[ij1]
    aezpt = wR*aez[ij] + wL*aez[ij1]
    abypt = wR*aby[ij] + wL*aby[ij1]
    abzpt = wR*abz[ij] + wL*abz[ij1]

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

    vx_out = np.full_like(x_original, np.nan)
    vy_out = np.full_like(x_original, np.nan)
    vz_out = np.full_like(x_original, np.nan)
    gamma_out = np.full_like(x_original, np.nan)

    vx_out[valid] = vx
    vy_out[valid] = vy
    vz_out[valid] = vz
    gamma_out[valid] = gamma

    return vx_out, vy_out, vz_out, gamma_out, ake

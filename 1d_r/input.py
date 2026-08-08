import numpy as np
from params import nx, nptcl, vth, vthi, qme, wce0, theta, seed, ij, par

np.random.seed(seed)
# ====================================
# Position at t = 0 For Electron
# ====================================
pos = np.empty(nptcl)
k = 0
for j in range(nx):
    n = par[j]
    i = ij[j]
    pos[k:k+n] = i + np.random.rand(n)
    k += n

# ====================================
# Velocity at t = 0 For Electron
# ====================================
vx0 = np.random.randn(nptcl)*vth
vy0 = np.random.randn(nptcl)*vth
vz0 = np.random.randn(nptcl)*vth
gamma0 = 1. / np.sqrt(1 - vx0*vx0 - vy0*vy0 - vz0*vz0)
ake0 = np.sum(gamma0 - 1)

# ====================================
# Position at t = 0 For Ion
# ====================================
pos_i = np.empty(nptcl)
k = 0
for j in range(nx):
    n = par[j]
    i = ij[j]
    pos_i[k:k+n] = i + np.random.rand(n)
    k += n
vxi0 = np.random.randn(nptcl)*vthi
vyi0 = np.random.randn(nptcl)*vthi
vzi0 = np.random.randn(nptcl)*vthi
gammai0 = 1. / np.sqrt(1 - vxi0*vxi0 - vyi0*vyi0 - vzi0*vzi0)
aki0 = np.sum(gammai0 - 1)

# ====================================
# Background Magneticfield
# ====================================
bx0 = wce0/qme*np.cos(theta/180.*np.pi)
bz0 = wce0/qme*np.sin(theta/180.*np.pi)

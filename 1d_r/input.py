import numpy as np
from params import nptcl, vth, vthi, qme, wce0, theta, seed, \
                    xmin, xmax, dx, x0
# import matplotlib.pyplot as plt

np.random.seed(seed)
# ====================================
# Position at t = 0 For Electron
# ====================================
i = np.arange(nptcl)
u_e = np.random.rand(nptcl)
x_ini = np.sqrt(xmin**2 + u_e * (xmax**2 - xmin**2))/dx
n = ((x_ini >= 0) & (x_ini < x0 + 1))
N = np.sum(n)

print(f'Number of particles in the most inner domain: {N} / {nptcl}')

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
i = np.arange(nptcl)
u_i = np.random.rand(nptcl)
xi_ini = np.sqrt(xmin**2 + u_i * (xmax**2 - xmin**2))/dx
ni = ((xi_ini >= 0) & (xi_ini < x0 + 1))
Ni = np.sum(ni)

print(f'Number of particles in the most inner domain: {Ni} / {nptcl}')
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

# plt.plot(np.sort(x*dx), np.arange(nptcl))
# plt.plot(np.sort(xi*dx), np.arange(nptcl))
# plt.xlabel('$r$', fontsize=15)
# plt.ylabel('$N(<=r)$', fontsize=15)
# plt.tick_params(labelsize=15)
# plt.show()

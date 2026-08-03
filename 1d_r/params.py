import numpy as np

# ===============================
# Simulation Setting
# ===============================
# nptcl = 4096*4               # Particle Number
xmin = 2**5
xmax = 2**6                       # Plasma Size Normalized By c/wpe
nx = 2**8                       # X-space Grid Number
nt = 2**8                    # Time Grid Number
dx = (xmax - xmin) / nx         # X-space Grid Length
dt = dx                         # Time Grid Length
theta = 90                       # Propagation Degree

r_edge = xmin + np.arange(nx+1)*dx
N0 = 100                        # most inner cell particle number
k = N0 / (r_edge[1]**2 - r_edge[0]**2)
ij = np.round(k * (r_edge[1:]**2 - r_edge[:-1]**2)).astype(int)
nptcl = ij.sum()

# ===============================
# Electron
# ===============================
wce0 = -0.5                     # Gyro Frequency
wpe = 1.0                       # Plasma Frequency c=wpe=e/m0=0
vth = 0.1                       # Thermal Speed
qme = -1.0                      # Electron Standard q/me=1
qe = (xmax - xmin)*wpe*wpe/(nptcl*qme)   # Electric Charge of an Electron
qdx = qe/dx                     # Electric Charge Density

# ===============================
# Ion
# ===============================
mi = 10                         # Ion Mass mi/me
wci0 = - wce0/mi                # Gyro Frequency
wpi = wpe/np.sqrt(mi)           # Plasma Frequency
vthi = vth/np.sqrt(mi)          # Thermal Speed
qmi = - qme/mi                  # Electron Standard q/mi
qi = - qe                       # Electric Charge
qidx = qi/dx                    # Electric Charge Density


# ===============================
# Others
# ===============================
# save_path = r'C:\Users\kasik\OneDrive - Kyushu University\PIC\Result'
# save_path = r'C:\Users\shimooka\OneDrive - Kyushu University (1)\PIC\Result'
save_path = r'C:\Users\kasik\OneDrive - Kyushu University\PIC\Result'
# save_path = './Result'
flag = True
bounds = [-5, -2]
seed = 0
dt_skip = 10
fps = 10
interval = 50
fontsize = 18
s = 1
bins = 100

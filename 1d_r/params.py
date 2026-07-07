import numpy as np

# ===============================
# Simulation Setting
# ===============================
# nptcl = 4096*4                  # Particle Number
xmax = 6                    # Plasma Size Normalized By c/wpe
# nx = 128                      # X-space Grid Number
nx = 128
k = 1
ij = 2*np.arange(nx) + 1
nptcl = ij.sum() * k
nt = 256*10                    # Time Grid Number
dx = xmax / nx                  # X-space Grid Length
dt = dx                         # Time Grid Length
theta = 0                       # Propagation Degree

# ===============================
# Electron
# ===============================
wce0 = -1.0                     # Gyro Frequency
wpe = 1.0                       # Plasma Frequency c=wpe=e/m0=0
vth = 0.1                       # Thermal Speed
qme = -1.0                      # Electron Standard q/me=1
qe = xmax*wpe*wpe/(nptcl*qme)   # Electric Charge of an Electron
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
flag = True
bounds = [-5, -2]
seed = 0
dt_skip = 10
fps = 10
interval = 50
fontsize = 18
s = 1
bins = 100

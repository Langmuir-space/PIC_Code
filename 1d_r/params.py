import numpy as np

# ===============================
# Simulation Setting
# ===============================
xmin = 2**0
xmax = 2**3                       # Plasma Size Normalized By c/wpe
nt = 2**11                    # Time Grid Number
dx = 2**(-3)
nx = int((xmax - xmin) / dx)        # X-space Grid Length
dt = dx                         # Time Grid Length
theta = 90                       # Propagation Degree
x0 = int(xmin / dx)
nptcl = 2**13                    # Particle Number


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
bounds = [None, None]
seed = 0
dt_skip = 10
fps = 10
interval = 50
fontsize = 18
s = 1
bins = 100

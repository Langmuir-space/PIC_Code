import numpy as np

# ===============================
# Simulation Setting
# ===============================
xmin = 2**15/10
xmax = 2**15/10 + 3                      # Plasma Size Normalized By c/wpe
nt = 2**10                    # Time Grid Number
# dx = 2**(-3)
dx = 0.2
nx = int((xmax - xmin) / dx)        # X-space Grid Length
dt = dx                         # Time Grid Length
theta = 90                       # Propagation Degree
x0 = int(xmin / dx)
nptcl = 2**14                    # Particle Number

print(xmin, xmax)

# ===============================
# Electron
# ===============================
wce0 = -1.0                     # Gyro Frequency
wpe = 1.0                       # Plasma Frequency c=wpe=e/m0=0
vth = 0.1                       # Thermal Speed
qme = -1.0                      # Electron Standard q/me=1
qe = np.pi*(xmax**2 - xmin**2)*wpe*wpe/(nptcl*qme)  # Electric Charge of an Electron
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

print(wce0**2 * wpi**2 / (wpe**2 * wce0**2))
print(wpe**2 + wce0**2)


# ===============================
# Others
# ===============================
# save_path = r'C:\Users\kasik\OneDrive - Kyushu University\PIC\Result'
# save_path = r'C:\Users\shimooka\OneDrive - Kyushu University (1)\PIC\Result'
save_path = r'C:\Users\kasik\OneDrive - Kyushu University\PIC\Result'
# save_path = './Result'
flag = True
bounds = [-6, -2]
seed = 0
dt_skip = 10
fps = 10
interval = 50
fontsize = 18
s = 1
bins = 100

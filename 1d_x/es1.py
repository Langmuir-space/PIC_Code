import numpy as np
import matplotlib.pyplot as plt

#--- input parameters ---
nptcl=4096; nx=128; xmax=12.8; nt=256
nxh=nx//2; dx=xmax/nx; dt=dx
bx=1.; by=np.zeros(nx); bz=np.zeros(nx) # B field
ex=np.zeros(nx); ey=np.zeros(nx); ez=np.zeros(nx) # E field
vth=0.1 # thermal velocity
#--- initial condisions ---
x =np.linspace(0, nx, nptcl, endpoint=False)
vx=np.random.randn(nptcl)*vth    # initial velocity in x-direction
vy=np.random.randn(nptcl)*vth    # initial velocity in y-direction
vz=np.random.randn(nptcl)*vth    # initial velocity in z-direction
gamma=1./np.sqrt(1.-vx*vx-vy*vy-vz*vz) # initial Lorentz factor
ake=np.sum(gamma-1.)  # initial kinetic energy

wpe=1.; qme=-1.  # plasma freq. and q/me
qe=xmax*wpe*wpe/(nptcl*qme) # electric charge of an electron

#-- subroutine move --
def move(vx,vy,vz,gamma,ae,abx):
    aex=ae*ex; aey=ae*ey; aez=ae*ez
    aby=ae*by; abz=ae*bz; ake=0.

    ij = np.floor(x).astype(int)
    ij1 = (ij + 1) % nx
    delx = x - ij
    aexpt = aex[ij] + delx * (aex[ij1] - aex[ij])
    aeypt = aey[ij] + delx * (aey[ij1] - aey[ij])
    aezpt = aez[ij] + delx * (aez[ij1] - aez[ij])
    abypt = aby[ij] + delx * (aby[ij1] - aby[ij])
    abzpt = abz[ij] + delx * (abz[ij1] - abz[ij])

    gvxs = vx * gamma + aexpt
    gvys = vy * gamma + aeypt
    gvzs = vz * gamma + aezpt

    gamma_temp = np.sqrt(1 + gvxs**2 + gvys**2 + gvzs**2)
    ake = np.sum(gamma_temp - 1)
    dcg = 1 / gamma_temp

    abxpt = dcg * abx; abypt = dcg * abypt; abzpt = dcg * abzpt
    vvx = gvxs + gvys * abzpt - gvzs * abypt
    vvy = gvys + gvzs * abxpt - gvxs * abzpt
    vvz = gvzs + gvxs * abypt - gvys * abxpt

    f = 2 / (1 + abxpt**2 + abypt**2 + abzpt**2)
    abxpt *= f; abypt *= f; abzpt *= f
    gvxs += vvy * abzpt - vvz * abypt + aexpt
    gvys += vvz * abxpt - vvx * abzpt + aeypt
    gvzs += vvx * abypt - vvy * abxpt + aezpt

    gamma = np.sqrt(1 + gvxs**2 + gvys**2 + gvzs**2)
    vx = gvxs / gamma
    vy = gvys / gamma
    vz = gvzs / gamma

    return vx,vy,vz,gamma,ake
#-- End of move --

#-- subroutine setrho --
qdx=qe/dx
def setrho():
    rho0=np.zeros(nx)

    ij = np.floor(x).astype(int)
    delx = x - ij
    ij1 = (ij + 1) % nx  # periodic boundary

    np.add.at(rho0, ij,  (1 - delx) * qdx)
    np.add.at(rho0, ij1, delx * qdx)
    rho0-=np.mean(rho0)
    return rho0
#-- End of setrho --

#-- subroutine field --
k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
rho=np.zeros(nx); aa=np.zeros(nx);
bb=np.sin(k*dx)/dx
aa[1:nx]=(dx/2/np.sin(k[1:nx]*dx/2))**2
def field(rho):
    rhok=np.fft.fft(rho)
    phik = aa*rhok
    ek=-1j*bb*phik
    ex[:]=np.real(np.fft.ifft(ek))
    return ex
#-- End of field --

#-- velocity at t=-dt/2
ae=0.5*qme*(-dt/2); abx=ae*bx
vx,vy,vz,gamma,ake=move(vx,vy,vz,gamma,ae,abx)

plt.ion()
fig=plt.figure(figsize=(16,4))
xx=np.linspace(0, xmax, nx)

rho=setrho()
ae=0.5*qme*dt; abx=ae*bx
ext=np.zeros((nt,nx))
for it in range(0,nt):
    vx,vy,vz,gamma,ake=move(vx,vy,vz,gamma,ae,abx)
    x+=vx*0.5
    x[x >= xmax/dx] -= xmax/dx
    x[x < 0] += xmax/dx

    plt.clf()
    img1=plt.subplot(141)
    img1.scatter(vx,vy,color='black',marker='o')
    plt.xlim(-0.5,0.5)
    plt.ylim(-0.5,0.5)
    plt.xlabel('$v_x(/c)$',fontsize=16)
    plt.ylabel('$v_y(/c)$',fontsize=16)

    img2=plt.subplot(142)
    img2.scatter(x*dx,vx,color='black',marker='o')
    plt.xlim(0,xmax)
    plt.ylim(-0.5,0.5)
    plt.xlabel('$x(*\omega_{pe}/c)$',fontsize=16)
    plt.ylabel('$v_x(/c)$',fontsize=16)

    img3=plt.subplot(143)
    img3.plot(xx,ex)
    plt.xlim(0,xmax)
    plt.ylim(-0.05,0.05)
    plt.xlabel('$x(*\omega_{pe}/c)$',fontsize=16)
    plt.ylabel('$E_x$',fontsize=16)

    img4=plt.subplot(144)
    img4.plot(xx,rho)
    plt.xlim(0,xmax)
    plt.ylim(-0.5,0.5)
    plt.xlabel('$x(*\omega_{pe}/c)$',fontsize=16)
    plt.ylabel('$rho$',fontsize=16)

    plt.pause(0.01)
    plt.tight_layout()

    x+=vx*0.5
    x[x >= xmax/dx] -= xmax/dx
    x[x < 0] += xmax/dx

    rho=setrho()
    ex=field(rho)
    ext[it,:]=ex[:]

fft_fld = np.fft.fft2(ext)
fft_shifted = np.fft.fftshift(fft_fld)
amp = np.abs(fft_shifted)

plt.clf()
img1=plt.subplot(141)
img1.scatter(vx,vy,color='black',marker='o')
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.xlabel('$v_x(/c)$',fontsize=16)
plt.ylabel('$v_y(/c)$',fontsize=16)

img2=plt.subplot(142)
img2.scatter(x*dx,vx,color='black',marker='o')
plt.xlim(0,xmax)
plt.ylim(-0.5,0.5)
plt.xlabel('$x(*\omega_{pe}/c)$',fontsize=16)
plt.ylabel('$v_x(/c)$',fontsize=16)

img3=plt.subplot(143)
img3.imshow(ext,interpolation='none',origin='lower',\
                            cmap='jet',aspect='auto')
plt.xlim(0,nx)
plt.ylim(0,nt)
plt.xlabel('$ix$',fontsize=16)
plt.ylabel('$it$',fontsize=16)

img4=plt.subplot(144)
img4.imshow(amp,interpolation='none',origin='lower',\
                            cmap='jet',aspect='auto')
plt.xlim(nxh-nx//4,nxh+nx//4)
plt.ylim(nt//2-32,nt//2+32)
plt.xlabel('$ik$',fontsize=16)
plt.ylabel('$iw$',fontsize=16)

plt.tight_layout() # avoid overlapping each figure
plt.ioff()
plt.show()
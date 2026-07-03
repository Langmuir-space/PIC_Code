import numpy as np
from params import nx, nt, nptcl, vth, vthi, qme, qmi, \
    dt, qdx, qidx, wce0, theta, seed, xmax, wpe, dx, qe, qi
from move import move
from current import curnt
from fields import field, field_energy
from setrho import setrho
from utils import make_dic
from viz import dispersion_plot, field_plot, phase_speed, animation
import time
import params


def main():
    np.random.seed(seed)
    # ====================================
    # Initial condition For Electron
    # ====================================
    x = np.linspace(0, nx, nptcl, endpoint=False)   # Position at t = 0
    vx = np.random.randn(nptcl)*vth                 # Velocity at t = Δt/2
    vy = np.random.randn(nptcl)*vth
    vz = np.random.randn(nptcl)*vth
    gamma = 1. / np.sqrt(1 - vx*vx - vy*vy - vz*vz)
    ake = np.sum(gamma - 1)                         # Particle Kinetic Energy

    # ====================================
    # Initial condition For Ion
    # ====================================
    xi = np.linspace(0, nx, nptcl, endpoint=False)
    vxi = np.random.randn(nptcl)*vthi
    vyi = np.random.randn(nptcl)*vthi
    vzi = np.random.randn(nptcl)*vthi
    gammai = 1. / np.sqrt(1 - vxi*vxi - vyi*vyi - vzi*vzi)
    aki = np.sum(gammai - 1)

    # ====================================
    # Background Magneticfield
    # ====================================
    bx0 = wce0/qme*np.cos(theta/180.*np.pi)
    bz0 = wce0/qme*np.sin(theta/180.*np.pi)

    # ====================================
    # Array Initialization
    # ====================================
    ex = np.zeros(nx)
    ey = np.zeros(nx)
    ez = np.zeros(nx)
    by = np.zeros(nx)
    bz = np.zeros(nx)
    eyl = np.zeros(nx)
    eyr = np.zeros(nx)
    ezl = np.zeros(nx)
    ezr = np.zeros(nx)

    # ====================================
    # Make Save Array
    # ====================================
    ext = []    # Raw electric field
    eyt = []
    ezt = []
    byt = []    # Raw magnetic field
    bzt = []
    xt = []    # Raw position
    vxt = []   # Raw velocity
    vyt = []
    vzt = []
    xit = []    # Raw position
    vxit = []   # Raw velocity
    vyit = []
    vzit = []
    phit = []

    aket = []         # Kinetic Energy of electron
    akit = []         # Kinetic Energy of ion
    ext2 = []         # Energy of Ex
    eyt2 = []
    ezt2 = []
    byt2 = []
    bzt2 = []
    rhoet = []
    rhoit = []

    xt.append(x.copy())
    xit.append(xi.copy())
    ext.append(ex.copy())
    eyt.append(ey.copy())
    ezt.append(ez.copy())
    byt.append(by.copy())
    bzt.append(bz.copy())
    vxt.append(vx.copy())
    vyt.append(vy.copy())
    vzt.append(vz.copy())
    vxit.append(vxi.copy())
    vyit.append(vyi.copy())
    vzit.append(vzi.copy())

    # ====================================
    # Velocity at t = -Δt/2 For Electron
    # ====================================
    ae = 0.5*qme*(-dt/2)
    tx = ae*bx0
    tz = ae*bz0
    vx, vy, vz, gamma, ake = move(
        vx, vy, vz, gamma, ae, tx, tz, x, ex, ey, ez, by, bz)

    # ====================================
    # Velocity at t = -Δt/2 For Ion
    # ====================================
    ai = 0.5*qmi*(-dt/2)
    txi = ai*bx0
    tzi = ai*bz0
    vxi, vyi, vzi, gammai, aki = move(
        vxi, vyi, vzi, gammai, ai, txi, tzi, xi, ex, ey, ez, by, bz)

    ae = 0.5*qme*dt
    tx = ae*bx0
    tz = ae*bz0

    ai = 0.5*qmi*dt
    txi = ai*bx0
    tzi = ai*bz0

    # ====================================
    # Make Save Dictionary
    # ====================================
    save_path = params.save_path
    flag = params.flag
    if flag:
        save_text_path, save_fig_path = make_dic(save_path)

    # ====================================
    # Time Step Loop
    # ====================================
    t0 = time.time()
    for it in range(0, nt):
        if it == 0:
            print("Calculation Start")

        if it % 10 == 0:
            elapsed = time.time() - t0
            print(f"Step {it}/{nt}  Elapsed: {elapsed:.2f} s")

        # =========================================
        # Calculate Velocity at t = (n + 1/2)Δt
        # Input:  v(n - 1/2)Δt, E(n), B(n)
        # Output: v(n + 1/2)Δt
        # =========================================

        vx, vy, vz, gamma, ake = move(
            vx, vy, vz, gamma, ae, tx, tz, x, ex, ey, ez, by, bz)

        vxi, vyi, vzi, gammai, aki = move(
            vxi, vyi, vzi, gammai, ai, txi, tzi, xi, ex, ey, ez, by, bz)

        # ===============================================================
        # Save Position at t = (n + 1/2)Δt
        # ===============================================================

        x_tmp = x + 0.5*vx
        x_tmp[x_tmp > nx] -= nx
        x_tmp[x_tmp < 0] += nx

        xi_tmp = xi + 0.5*vxi
        xi_tmp[xi_tmp > nx] -= nx
        xi_tmp[xi_tmp < 0] += nx

        xt.append(x_tmp.copy())
        xit.append(xi_tmp.copy())

        # ===============================================================
        # Calculate Current at t = (n + 1/2)Δt and Rho at t = (n + 1)Δt
        # Input: v(n + 1/2)Δt, x(n)Δt
        # Output: J±(n + 1/2)Δt, rho(n + 1)Δt, x(n + 1)Δt
        # ===============================================================

        jym_e, jzm_e, jyp_e, jzp_e, rho_e, x = \
            curnt(x, vx, vy, vz, qdx)
        jym_i, jzm_i, jyp_i, jzp_i, rho_i, xi = \
            curnt(xi, vxi, vyi, vzi, qidx)

        jym = jym_e + jym_i
        jzm = jzm_e + jzm_i
        jyp = jyp_e + jyp_i
        jzp = jzp_e + jzp_i
        rho = rho_e + rho_i

        # ======================================
        # Calculate Field at t = (n + 1)Δt
        # Input: J±(n + 1/2)Δt, rho(n + 1)Δt
        # Output: E(n + 1)Δt, B(n + 1)Δt
        # ======================================
        # ex, ey, ez, by, bz, eyl, eyr, ezl, ezr = field(
        #     jym, jzm, jyp, jzp, rho, eyl, eyr, ezl, ezr)

        # for electrostatic
        ex = field(jym, jzm, jyp, jzp, rho, eyl, eyr, ezl, ezr)
        # ex = setrho(rho)

        # ======================================
        # Save Into List
        # ======================================
        ex2, ey2, ez2, by2, bz2 = field_energy(ex, ey, ez, by, bz)
        ext.append(ex.copy())
        eyt.append(ey.copy())
        ezt.append(ez.copy())
        byt.append(by.copy())
        bzt.append(bz.copy())
        vxt.append(vx.copy())
        vyt.append(vy.copy())
        vzt.append(vz.copy())
        vxit.append(vxi.copy())
        vyit.append(vyi.copy())
        vzit.append(vzi.copy())
        # phit.append(phi.copy())
        rhoet.append((rho_e/qe).copy())
        rhoit.append((rho_i/qi).copy())

        aket.append(ake.copy())
        akit.append(aki.copy())
        ext2.append(ex2.copy())
        eyt2.append(ey2.copy())
        ezt2.append(ez2.copy())
        byt2.append(by2.copy())
        bzt2.append(bz2.copy())

    # ======================================
    # Convert List to Array
    # ======================================

    ext = np.array(ext)
    eyt = np.array(eyt)
    ezt = np.array(ezt)
    byt = np.array(byt)
    bzt = np.array(bzt)
    xt = np.array(xt)
    vxt = np.array(vxt)
    vyt = np.array(vyt)
    vzt = np.array(vzt)
    xit = np.array(xit)
    vxit = np.array(vxit)
    vyit = np.array(vyit)
    vzit = np.array(vzit)
    aket = np.array(aket)
    akit = np.array(akit)
    ext2 = np.array(ext2)
    eyt2 = np.array(eyt2)
    ezt2 = np.array(ezt2)
    byt2 = np.array(byt2)
    bzt2 = np.array(bzt2)
    rhoet = np.array(rhoet)
    rhoit = np.array(rhoit)

    # ======================================
    # Make Animation and Save Figures
    # ======================================

    """
    animation(vxt, vyt, save_name=f"{save_fig_path}/vx-vy.gif",
              xlabel='$v_{xe}(/c)$', ylabel='$v_{ye}(/c)$',
              xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5)
    animation(xt, vxt, save_name=f"{save_fig_path}/x-vy.gif",
              xlabel='$x_e(*\\Omega_{pe}/c)$', ylabel='$v_{xe}(/c)$',
              xmin=0, xmax=None, ymin=-0.5, ymax=0.5)
    animation(vxit, vyit, save_name=f"{save_fig_path}/vxi-vyi.gif",
              xlabel='$v_{xi}(/c)$', ylabel='$v_{yi}(/c)$',
              xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5)
    animation(xit, vxit, save_name=f"{save_fig_path}/xi-vxi.gif",
              xlabel='$x_i(*\\Omega_{pe}/c)$', ylabel='$v_{xi}(/c)$',
              xmin=0, xmax=None, ymin=-0.5, ymax=0.5)
    """

    # dispersion_plot(ext, save_fig_path, title=r'$E_x(k,\omega)$', label='Ex_wk')
    # dispersion_plot(eyt, save_fig_path, title=r'$E_y(k,\omega)$', label='Ey_wk')
    # dispersion_plot(ezt, save_fig_path, title=r'$E_z(k,\omega)$', label='Ez_wk')
    # dispersion_plot(byt, save_fig_path, title=r'$B_y(k,\omega)$', label='By_wk')
    # dispersion_plot(bzt, save_fig_path, title=r'$B_z(k,\omega)$', label='Bz_wk')

    # field_plot(ext, save_fig_path, title=r'$E_x(x,t)$', label='Ex_xt')
    # field_plot(eyt, save_fig_path, title=r'$E_y(x,t)$', label='Ey_xt')
    # field_plot(ezt, save_fig_path, title=r'$E_z(x,t)$', label='Ez_xt')
    # field_plot(byt, save_fig_path, title=r'$B_y(x,t)$', label='By_xt')
    # field_plot(bzt, save_fig_path, title=r'$B_z(x,t)$', label='Bz_xt')

    # phase_speed(vx, vy, save_fig_path, title='Electron phase space',
    #             label='Electron_phase', vmin=None, vmax=None)
    # phase_speed(vxi, vyi, save_fig_path, title='Ion phase space',
    #             label='Ion_phase', vmin=None, vmax=None)

    ij = np.arange(nx)*dx
    # animation(ij, rhoet, save_name=f"{save_fig_path}/rho_e.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$\\rho_e$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(ij, rhoit, save_name=f"{save_fig_path}/rho_i.gif",
    #           xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$\\rho_i$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(ij, ext, save_name=f"{save_fig_path}/ex.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$E_{x}$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(ij, phit, save_name=f"{save_fig_path}/phi.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$\\phi$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(vxt, vxt, save_name=f"{save_fig_path}/vx_f.gif",
    #           xlabel='$v_{xe}(/c)$', ylabel='$f(v_{xe})$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='hist')
    # animation(vxt, vyt, save_name=f"{save_fig_path}/vx-vy.gif",
    #           xlabel='$v_{xe}(/c)$', ylabel='$v_{ye}(/c)$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='phase')
    # animation(xt*dx, vxt, save_name=f"{save_fig_path}/x-vy.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$v_{xe}(/c)$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='phase')
    # animation(vxit, vyit, save_name=f"{save_fig_path}/vxi-vyi.gif",
    #           xlabel='$v_{xi}(/c)$', ylabel='$v_{yi}(/c)$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='phase')
    # animation(xit*dx, vxit, save_name=f"{save_fig_path}/xi-vxi.gif",
    #           xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$v_{xi}(/c)$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='phase')

    dispersion_plot(ext, save_fig_path, title=r'$E_x(k,\omega)$', label='Ex_wk')
    dispersion_plot(eyt, save_fig_path, title=r'$E_y(k,\omega)$', label='Ey_wk')
    dispersion_plot(ezt, save_fig_path, title=r'$E_z(k,\omega)$', label='Ez_wk')
    dispersion_plot(byt, save_fig_path, title=r'$B_y(k,\omega)$', label='By_wk')
    dispersion_plot(bzt, save_fig_path, title=r'$B_z(k,\omega)$', label='Bz_wk')

    # phase_speed(vx, vy, save_fig_path, title='Electron phase space',
    #             label='Electron_phase', vmin=None, vmax=None)
    # phase_speed(vxi, vyi, save_fig_path, title='Ion phase space',
    #             label='Ion_phase', vmin=None, vmax=None)

    field_plot(ext, save_fig_path, title=r'$E_x(x,t)$', label='Ex_xt')
    # field_plot(eyt, save_fig_path, title=r'$E_y(x,t)$', label='Ey_xt')
    # field_plot(ezt, save_fig_path, title=r'$E_z(x,t)$', label='Ez_xt')
    # field_plot(byt, save_fig_path, title=r'$B_y(x,t)$', label='By_xt')
    # field_plot(bzt, save_fig_path, title=r'$B_z(x,t)$', label='Bz_xt')


if __name__ == "__main__":
    main()

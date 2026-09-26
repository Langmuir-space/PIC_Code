import numpy as np
from params import nx, nt, qme, qmi, dt, qe, qi, save_path, flag, dx
from move import move, push
from fields import field_energy, field_ex, field
from utils import make_dic
from viz import field_plot, animation, dispersion_plot, phase_speed
from setrho import setrho, index
import time
from input import x_ini, vx0, vy0, vz0, xi_ini, vxi0, vyi0, vzi0, \
    gamma0, gammai0, ake0, aki0, bx0, bz0
from boundary import ptcle_bc
from current import curnt


def main():
    # ====================================
    # Load Initial Condition
    # ====================================
    x, vx, vy, vz, xi, vxi, vyi, vzi, gamma, gammai, ake, aki = \
        x_ini, vx0, vy0, vz0, xi_ini, vxi0, vyi0, vzi0, gamma0, gammai0, ake0, aki0

    # ====================================
    # Array Initialization
    # ====================================
    ex = np.zeros(nx + 1)
    ey = np.zeros(nx + 1)
    ez = np.zeros(nx + 1)
    by = np.zeros(nx + 1)
    bz = np.zeros(nx + 1)

    # ====================================
    # Fields at t = 0
    # ====================================
    rhoe = setrho(x, qe)
    rhoi = setrho(xi, qi)
    rho = rhoe + rhoi
    ex, phi = field_ex(rho)
    ex2, ey2, ez2, by2, bz2 = field_energy(ex, ey, ez, by, bz)

    # ======================================================
    # Make Save Array at t = nΔt (veloity at t = (n-1/2)Δt)
    # ======================================================
    save = {}
    for name in [
        "ex", "ey", "ez", "by", "bz", "x", "vx", "vy", "vz",
        "xi", "vxi", "vyi", "vzi", "phi", "ake", "aki",
        "ex2", "ey2", "ez2", "by2", "bz2", "rhoe", "rhoi", "rhoei"
            ]:
        save[name] = []

    # ====================================
    # Velocity at t = -Δt/2
    # ====================================
    ae = 0.5*qme*(-dt/2); tx = ae*bx0; tz = ae*bz0
    vx, vy, vz, gamma, ake = move(
        vx, vy, vz, gamma, ae, tx, tz, x, ex, ey, ez, by, bz)

    ai = 0.5*qmi*(-dt/2); txi = ai*bx0; tzi = ai*bz0
    vxi, vyi, vzi, gammai, aki = move(
        vxi, vyi, vzi, gammai, ai, txi, tzi, xi, ex, ey, ez, by, bz)

    ae = 0.5*qme*dt; tx = ae*bx0; tz = ae*bz0
    ai = 0.5*qmi*dt; txi = ai*bx0; tzi = ai*bz0

    # ===========================================================
    # Save positions and fields at t = 0 (veloity at t = -Δt/2)
    # ===========================================================
    save_vars = {"x": x, "xi": xi, "ex": ex, "ey": ey, "ez": ez,
                 "by": by, "bz": bz, "phi": phi, "vx": vx, "vy": vy,
                 "vz": vz, "vxi": vxi, "vyi": vyi, "vzi": vzi, "rhoe": rhoe,
                 "rhoi": rhoi, "rhoei": rho, "ake": ake, "aki": aki,
                 "ex2": ex2, "ey2": ey2, "ez2": ez2, "by2": by2, "bz2": bz2}

    for key, value in save_vars.items():
        save[key].append(value.copy())

    # ====================================
    # Make Save Dictionary
    # ====================================
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
        # Velocity at t = (n + 1/2)Δt
        # =========================================
        vx, vy, vz, gamma, ake = move(
            vx, vy, vz, gamma, ae, tx, tz, x, ex, ey, ez, by, bz)

        vxi, vyi, vzi, gammai, aki = move(
            vxi, vyi, vzi, gammai, ai, txi, tzi, xi, ex, ey, ez, by, bz)

        # ===============================================================
        # Push Particle Position at t = (n + 1)Δt
        # ===============================================================
        x_old = x.copy()
        vx_old = vx.copy()
        vy_old = vy.copy()
        vz_old = vz.copy()
        x, vx, vy = push(x_old, vx_old, vy_old)

        xi_old = xi.copy()
        vxi_old = vxi.copy()
        vyi_old = vyi.copy()
        vzi_old = vzi.copy()
        xi, vxi, vyi = push(xi_old, vxi_old, vyi_old)

        # ===============================================================
        # Particle Boundary Conditions at t = (n + 1)Δt
        # ===============================================================
        x, vx, vy, vz = ptcle_bc(x, vx, vy, vz)
        xi, vxi, vyi, vzi = ptcle_bc(xi, vxi, vyi, vzi)

        # ===============================================================
        # Current density at t = (n + 1/2)Δt at x = i
        # ===============================================================
        jye_old, jze_old = curnt(x_old, vy_old, vz_old, qe)
        jye_new, jze_new = curnt(x, vy, vz, qe)

        jyi_old, jzi_old = curnt(xi_old, vyi_old, vzi_old, qi)
        jyi_new, jzi_new = curnt(xi, vyi, vzi, qi)

        jy_old = jye_old + jyi_old
        jz_old = jze_old + jzi_old
        jy_new = jye_new + jyi_new
        jz_new = jze_new + jzi_new
        jy = 0.5*(jy_old + jy_new)
        jz = 0.5*(jz_old + jz_new)

        # ===============================================================
        # Charge Density at t = (n + 1/2)Δt at x = i
        # ===============================================================
        rhoe = setrho(x, qe)
        rhoi = setrho(xi, qi)
        rho = rhoe + rhoi

        # ======================================
        # Field at t = (n + 1)Δt
        # ======================================
        # ex, ey, ez, by, bz = field(jy, jz, rho)
        ex, phi = field_ex(rho)      # for electrostatic

        # ======================================
        # Save Into List
        # ======================================
        ex2, ey2, ez2, by2, bz2 = field_energy(ex, ey, ez, by, bz)
        save_vars = {"x": x, "xi": xi, "ex": ex, "ey": ey, "ez": ez,
                     "by": by, "bz": bz, "phi": phi, "vx": vx, "vy": vy,
                     "vz": vz, "vxi": vxi, "vyi": vyi, "vzi": vzi,
                     "rhoe": rhoe, "rhoi": rhoi, "rhoei": rho, "ake": ake,
                     "aki": aki, "ex2": ex2, "ey2": ey2, "ez2": ez2,
                     "by2": by2, "bz2": bz2}

        for key, value in save_vars.items():
            save[key].append(value.copy())

    # ======================================
    # Convert List to Array
    # ======================================
    save = {key: np.array(value) for key, value in save.items()}

    # ======================================
    # Make Animation and Save Figures
    # ======================================
    # animation(ij, vj*save["rhoe"]/qe, save_name=f"{save_fig_path}/N_e.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$N_e$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(ij, vj*save["rhoi"]/qi, save_name=f"{save_fig_path}/N_i.gif",
    #           xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$N_i$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(index*dx, save["rhoe"], save_name=f"{save_fig_path}/rho_e.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$\\rho_e$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(index*dx, save["rhoi"], save_name=f"{save_fig_path}/rho_i.gif",
    #           xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$\\rho_i$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(index*dx, save["rhoei"], save_name=f"{save_fig_path}/rho.gif",
    #           xlabel='$x(*\\omega_{pe}/c)$', ylabel='$\\rho$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    animation(index*dx, save["ex"], save_name=f"{save_fig_path}/ex.gif",
              xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$E_{x}$',
              xmin=None, xmax=None, ymin=None, ymax=None,
              select='raw')
    # animation(index*dx, save["phi"], save_name=f"{save_fig_path}/phi.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$\\phi$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # velocity_e = np.sqrt(save["vex"]**2 + save["vyt"]**2 + save["vzt"]**2)
    # animation(velocity_e, save["vxt"], save_name=f"{save_fig_path}/ve_f.gif",
    #           xlabel='$v_{e}(/c)$', ylabel='$f(v_{e})$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='hist')
    # velocity_i = np.sqrt(save["vxit"]**2 + save["vyit"]**2 + save["vzit"]**2)
    # animation(velocity_i, save["vxit"], save_name=f"{save_fig_path}/vi_f.gif",
    #           xlabel='$v_{i}(/c)$', ylabel='$f(v_{i})$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='hist')
    # animation(save["vxt"], save["vyt"], save_name=f"{save_fig_path}/vx-vy.gif",
    #           xlabel='$v_{xe}(/c)$', ylabel='$v_{ye}(/c)$',
    #           xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
    #           select='phase')
    # animation(save["xt"]*dx, save["vxt"], save_name=f"{save_fig_path}/x-vx.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$v_{xe}(/c)$',
    #           xmin=0, xmax=None, ymin=-0.5, ymax=0.5,
    #           select='phase')
    # animation(save["vxit"], save["vyit"], save_name=f"{save_fig_path}/vxi-vyi.gif",
    #           xlabel='$v_{xi}(/c)$', ylabel='$v_{yi}(/c)$',
    #           xmin=-0.25, xmax=0.25, ymin=-0.25, ymax=0.25,
    #           select='phase')
    # animation(save["xit"]*dx, save["vxit"], save_name=f"{save_fig_path}/xi-vxi.gif",
    #           xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$v_{xi}(/c)$',
    #           xmin=0, xmax=None, ymin=-0.25, ymax=0.25,
    #           select='phase')

    dispersion_plot(save["ex"], save_fig_path, title=r'$E_x(k,\omega)$',
                    label='Ex_wk')
    # dispersion_plot(save["ey"], save_fig_path, title=r'$E_y(k,\omega)$',
    #                 label='Ey_wk')
    # dispersion_plot(save["ez"], save_fig_path, title=r'$E_z(k,\omega)$',
    #                 label='Ez_wk')
    # dispersion_plot(save["by"], save_fig_path, title=r'$B_y(k,\omega)$',
    #                 label='By_wk')
    # dispersion_plot(save["bz"], save_fig_path, title=r'$B_z(k,\omega)$',
    #                 label='Bz_wk')

    phase_speed(vx, vy, save_fig_path, title='Electron phase space',
                label='Electron_phase', vmin=None, vmax=None)
    phase_speed(vxi, vyi, save_fig_path, title='Ion phase space',
                label='Ion_phase', vmin=None, vmax=None)

    field_plot(save["ex"], save_fig_path, title=r'$E_x(x,t)$', label='Ex_xt')
    # field_plot(save["ey"], save_fig_path, title=r'$E_y(x,t)$', label='Ey_xt')
    # field_plot(save["ez"], save_fig_path, title=r'$E_z(x,t)$', label='Ez_xt')
    # field_plot(save["by"], save_fig_path, title=r'$B_y(x,t)$', label='By_xt')
    # field_plot(save["bz"], save_fig_path, title=r'$B_z(x,t)$', label='Bz_xt')


if __name__ == "__main__":
    main()

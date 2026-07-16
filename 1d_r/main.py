import numpy as np
from params import nx, nt, qme, qmi, dt, qe, qi, dx
from move import move
from fields import field_energy, field_ex
from utils import make_dic
from viz import field_plot, animation
from setrho import setrho
import time
import params
from input import x0, vx0, vy0, vz0, xi0, vxi0, vyi0, vzi0, \
    gamma0, gammai0, ake0, aki0, bx0, bz0


def main():
    # ====================================
    # Load Initial Condition
    # ====================================
    x = x0
    vx = vx0
    vy = vy0
    vz = vz0
    xi = xi0
    vxi = vxi0
    vyi = vyi0
    vzi = vzi0
    gamma = gamma0
    gammai = gammai0
    ake = ake0
    aki = aki0

    # ====================================
    # Array Initialization
    # ====================================
    ex = np.zeros(nx + 1)
    ey = np.zeros(nx + 1)
    ez = np.zeros(nx + 1)
    by = np.zeros(nx + 1)
    bz = np.zeros(nx + 1)
    # eyl = np.zeros(nx + 1)
    # eyr = np.zeros(nx + 1)
    # ezl = np.zeros(nx + 1)
    # ezr = np.zeros(nx + 1)

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
        "ex2", "ey2", "ez2", "by2", "bz2", "rhoe", "rhoi"
            ]:
        save[name] = []

    # ====================================
    # Velocity at t = -Δt/2
    # ====================================
    ae = 0.5*qme*(-dt/2)
    tx = ae*bx0
    tz = ae*bz0
    vx, vy, vz, gamma, ake = move(
        vx, vy, vz, gamma, ae, tx, tz, x, ex, ey, ez, by, bz)

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

    # ===========================================================
    # Save positions and fields at t = 0 (veloity at t = -Δt/2)
    # ===========================================================
    save_vars = {
        "x": x,
        "xi": xi,
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "by": by,
        "bz": bz,
        "phi": phi,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "vxi": vxi,
        "vyi": vyi,
        "vzi": vzi,
        "rhoe": rhoe,
        "rhoi": rhoi,
        "ake": ake,
        "aki": aki,
        "ex2": ex2,
        "ey2": ey2,
        "ez2": ez2,
        "by2": by2,
        "bz2": bz2,
    }

    for key, value in save_vars.items():
        save[key].append(value.copy())

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
        # Push Particle Position at t = (n + 1)Δt
        # ===============================================================
        x2 = x + vx
        y2 = vy
        r2 = np.sqrt(x2**2 + y2**2)
        alpha = np.arctan2(y2, x2)
        x = r2
        # th += alpha
        vx_old = vx.copy()
        vy_old = vy.copy()
        vx = np.cos(alpha)*vx_old + np.sin(alpha)*vy_old
        vy = -np.sin(alpha)*vx_old + np.cos(alpha)*vy_old

        mask_out = x >= nx
        x[mask_out] = np.nan
        vx[mask_out] = np.nan
        vy[mask_out] = np.nan
        vz[mask_out] = np.nan
        gamma[mask_out] = np.nan

        x2 = xi + vxi
        y2 = vyi
        r2 = np.sqrt(x2**2 + y2**2)
        alpha = np.arctan2(y2, x2)
        xi = r2
        # th += alpha
        vxi_old = vxi.copy()
        vyi_old = vyi.copy()
        vxi = np.cos(alpha)*vxi_old + np.sin(alpha)*vyi_old
        vyi = -np.sin(alpha)*vxi_old + np.cos(alpha)*vyi_old

        mask_out = xi >= nx
        xi[mask_out] = np.nan
        vxi[mask_out] = np.nan
        vyi[mask_out] = np.nan
        vzi[mask_out] = np.nan
        gammai[mask_out] = np.nan

        # ===============================================================
        # Calculate Current at t = (n + 1/2)Δt and Rho at t = (n + 1)Δt
        # Input: v(n + 1/2)Δt, x(n)Δt
        # Output: J±(n + 1/2)Δt, rho(n + 1)Δt, x(n + 1)Δt
        # ===============================================================

        # jym_e, jzm_e, jyp_e, jzp_e, rhoe, x = \
        #     curnt(x, vx, vy, vz, qe)
        # jym_i, jzm_i, jyp_i, jzp_i, rhoi, xi = \
        #     curnt(xi, vxi, vyi, vzi, qi)

        # jym = jym_e + jym_i
        # jzm = jzm_e + jzm_i
        # jyp = jyp_e + jyp_i
        # jzp = jzp_e + jzp_i
        rhoe = setrho(x, qe)
        rhoi = setrho(xi, qi)
        rho = rhoe + rhoi

        # ======================================
        # Calculate Field at t = (n + 1)Δt
        # Input: J±(n + 1/2)Δt, rho(n + 1)Δt
        # Output: E(n + 1)Δt, B(n + 1)Δt
        # ======================================
        # ex, ey, ez, by, bz, eyl, eyr, ezl, ezr = field(
        #     jym, jzm, jyp, jzp, rho, eyl, eyr, ezl, ezr)

        ex, phi = field_ex(rho)      # for electrostatic

        # ======================================
        # Save Into List
        # ======================================
        ex2, ey2, ez2, by2, bz2 = field_energy(ex, ey, ez, by, bz)
        save_vars = {
            "x": x,
            "xi": xi,
            "ex": ex,
            "ey": ey,
            "ez": ez,
            "by": by,
            "bz": bz,
            "phi": phi,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "vxi": vxi,
            "vyi": vyi,
            "vzi": vzi,
            "rhoe": rhoe,
            "rhoi": rhoi,
            "ake": ake,
            "aki": aki,
            "ex2": ex2,
            "ey2": ey2,
            "ez2": ez2,
            "by2": by2,
            "bz2": bz2,
        }

        for key, value in save_vars.items():
            save[key].append(value.copy())

    # ======================================
    # Convert List to Array
    # ======================================
    ext = np.array(save["ex"])
    # eyt = np.array(save["ey"])
    # ezt = np.array(save["ez"])
    # byt = np.array(save["by"])
    # bzt = np.array(save["bz"])
    # xt = np.array(save["x"])
    # vxt = np.array(save["vx"])
    # vyt = np.array(save["vy"])
    # vzt = np.array(save["vz"])
    # xit = np.array(save["xi"])
    # vxit = np.array(save["vxi"])
    # vyit = np.array(save["vyi"])
    # vzit = np.array(save["vzi"])
    # aket = np.array(save["ake"])
    # akit = np.array(save["aki"])
    # ext2 = np.array(save["ex2"])
    # eyt2 = np.array(save["ey2"])
    # ezt2 = np.array(save["ez2"])
    # byt2 = np.array(save["by2"])
    # bzt2 = np.array(save["bz2"])
    # phit = np.array(save["phi"])
    rhoet = np.array(save["rhoe"])
    rhoit = np.array(save["rhoi"])

    # ======================================
    # Make Animation and Save Figures
    # ======================================
    ij = np.arange(nx+1)*dx
    # animation(ij, vj*rhoet/qe, save_name=f"{save_fig_path}/N_e.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$N_e$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(ij, vj*rhoit/qi, save_name=f"{save_fig_path}/N_i.gif",
    #           xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$N_i$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    animation(ij, rhoet/qe, save_name=f"{save_fig_path}/n_e.gif",
              xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$n_e$',
              xmin=None, xmax=None, ymin=None, ymax=None,
              select='raw')
    animation(ij, rhoit/qi, save_name=f"{save_fig_path}/n_i.gif",
              xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$n_i$',
              xmin=None, xmax=None, ymin=None, ymax=None,
              select='raw')
    # animation(ij, ext, save_name=f"{save_fig_path}/ex.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$E_{x}$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # animation(ij, phit, save_name=f"{save_fig_path}/phi.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$\\phi$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='raw')
    # velocity_e = np.sqrt(vxt**2 + vyt**2 + vzt**2)
    # animation(velocity_e, vxt, save_name=f"{save_fig_path}/ve_f.gif",
    #           xlabel='$v_{e}(/c)$', ylabel='$f(v_{e})$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='hist')
    # velocity_i = np.sqrt(vxit**2 + vyit**2 + vzit**2)
    # animation(velocity_i, vxit, save_name=f"{save_fig_path}/vi_f.gif",
    #           xlabel='$v_{i}(/c)$', ylabel='$f(v_{i})$',
    #           xmin=None, xmax=None, ymin=None, ymax=None,
    #           select='hist')
    # animation(vxt, vyt, save_name=f"{save_fig_path}/vx-vy.gif",
    #           xlabel='$v_{xe}(/c)$', ylabel='$v_{ye}(/c)$',
    #           xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
    #           select='phase')
    # animation(xt*dx, vxt, save_name=f"{save_fig_path}/x-vy.gif",
    #           xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$v_{xe}(/c)$',
    #           xmin=0, xmax=None, ymin=-0.5, ymax=0.5,
    #           select='phase')
    # animation(vxit, vyit, save_name=f"{save_fig_path}/vxi-vyi.gif",
    #           xlabel='$v_{xi}(/c)$', ylabel='$v_{yi}(/c)$',
    #           xmin=-0.25, xmax=0.25, ymin=-0.25, ymax=0.25,
    #           select='phase')
    # animation(xit*dx, vxit, save_name=f"{save_fig_path}/xi-vxi.gif",
    #           xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$v_{xi}(/c)$',
    #           xmin=0, xmax=None, ymin=-0.25, ymax=0.25,
    #           select='phase')

    # dispersion_plot(ext, save_fig_path, title=r'$E_x(k,\omega)$',
    #                 label='Ex_wk')
    # dispersion_plot(eyt, save_fig_path, title=r'$E_y(k,\omega)$',
    #                 label='Ey_wk')
    # dispersion_plot(ezt, save_fig_path, title=r'$E_z(k,\omega)$',
    #                 label='Ez_wk')
    # dispersion_plot(byt, save_fig_path, title=r'$B_y(k,\omega)$',
    #                 label='By_wk')
    # dispersion_plot(bzt, save_fig_path, title=r'$B_z(k,\omega)$',
    #                 label='Bz_wk')

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

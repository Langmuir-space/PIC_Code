import numpy as np
from params import nx, nt, qme, qmi, dt, qe, qi, save_path, flag, dx
from move import move, push
from fields import field_energy, field_ex, ftdt, convert
from utils import make_dic
from viz import save_field_plot, save_animation, save_phase_speed, save_dispersion
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
    byh = np.zeros(nx)
    bzh = np.zeros(nx)

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
        # Push and Update Particle Position and Velocity at t = (n + 1)Δt
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
        # Charge Density at t = (n + 1)Δt at x = i
        # ===============================================================
        rhoe = setrho(x, qe)
        rhoi = setrho(xi, qi)
        rho = rhoe + rhoi

        # ======================================
        # Field at t = (n + 1)Δt at x = i
        # ======================================
        ex, phi = field_ex(rho)
        ey, ez, byh, bzh = ftdt(ey, ez, byh, bzh, jy, jz)
        by, bz = convert(byh, bzh)

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
    save_animation(save, index, dx, save_fig_path)
    save_dispersion(save, index, dx, save_fig_path)
    save_phase_speed(vx, vy, vxi,vyi, save_fig_path)
    save_field_plot(save, save_fig_path)


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from params import dx, dt, xmax, wpe, dt_skip, fps, interval, fontsize, \
    s, bounds, bins, nt, xmin, nx, Tmax, wce0, wci0, Tmax
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from setrho import index
from disp import dispersion_relation, fluctuation_profile

n = 100
wmax = 2*np.pi*n/(nt*dt)


def dispersion_plot(field, fm, Tm, wt_m, save_path, title, label, flag):

    def amplitude(field, fm):
        norm = np.sum(index*fm**2, axis=1)
        nume = np.zeros((field.shape[0], fm.shape[0]))
        for i in range(field.shape[0]):
            for j in range(fm.shape[0]):
                nume[i,j] = np.sum(index*field[i, :] * fm[j])
        amp = nume / norm
        return amp

    amp = amplitude(field, fm)
    fft = np.fft.rfft(amp, axis=0)
    power = np.log10(np.abs(fft)/ amp.shape[0])
    w = np.fft.rfftfreq(amp.shape[0], d=dt)*2*np.pi

    fig, ax = plt.subplots(constrained_layout=True)

    pcm = ax.pcolormesh(Tm, w, power, shading='nearest')

    if flag:
        ax.scatter(Tm, wt_m, c='black', s=10, label='$\\xi_m$')
    elif not flag:
        ax.scatter(np.repeat(Tm, wt_m.shape[1]), wt_m.real.ravel(),
                    c='black', s=10, label='$\\eta_m$')

    ax.set_xlabel(r'$T_m c/\omega_{pe}$', fontsize=fontsize)
    ax.set_ylabel(r'$\omega/\omega_{pe}$', fontsize=fontsize)
    ax.set_title(rf'{title}', fontsize=fontsize)
    ax.set_xlim(0, Tmax)
    ax.set_ylim(0, wmax)
    ax.tick_params(axis='x', labelsize=fontsize - 2)
    ax.tick_params(axis='y', labelsize=fontsize - 2)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    fig.savefig(os.path.join(save_path, f"{label}.png"))
    plt.close(fig)


def theory_plot(xi, w_xi, xi_m, w_xi_m, eta, w_eta, eta_m, w_eta_m, save_path, label):

    plt.scatter(xi, w_xi, s=0.1, c='black')
    plt.scatter(xi_m, w_xi_m, c='r', label='$\\xi_m$')
    plt.scatter(np.repeat(eta, w_eta.shape[1]), w_eta.real.ravel(),
                s=0.1, c='black')
    plt.scatter(np.repeat(eta_m, w_eta_m.shape[1]), w_eta_m.real.ravel(),
                c='blue', label='$\\eta_m$')
    plt.xlim(0, None)
    plt.ylim(0, wmax)
    plt.xlabel('$\\xi, \\eta \\,(*c/\\omega_{pe})$', fontsize=15)
    plt.ylabel('$\\omega / \\omega_{pe}$', fontsize=15)
    plt.title(f'$\\omega_{{ce}}/\\omega_{{pe}}={wce0}, \
            \\omega_{{ci}}/\\omega_{{pe}}={wci0}, \
            {xmin} \\leq r \\leq {xmax}$', fontsize=15)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{label}.png"))
    plt.close()


def field_plot(field, save_path, title, label):

    fig, ax = plt.subplots(constrained_layout=True)
    nt = field.shape[0]
    h = ax.imshow(field, extent=[xmin, xmax, 0, nt*dt], origin='lower',
                  cmap='jet', aspect='auto')

    ax.set_xlabel(r'$x\,(\omega_{pe}/c)$', fontsize=fontsize)
    ax.set_ylabel(r'$\omega_{pe} t$', fontsize=fontsize)
    ax.set_title(rf'{title}', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize - 2)
    ax.tick_params(axis='y', labelsize=fontsize - 2)

    cbar = fig.colorbar(h, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    fig.savefig(os.path.join(save_path, f"{label}.png"))
    plt.close(fig)


def phase_speed(vx, vy, save_path, title, label, vmin=None, vmax=None):

    if vmin is None:
        vmin = np.min(vx)
    if vmax is None:
        vmax = np.max(vx)

    fig, ax = plt.subplots(constrained_layout=True)

    h = ax.hist2d(vx, vy, bins=bins, range=[[vmin, vmax], [vmin, vmax]],
                  cmap='jet', norm='log')

    ax.set_xlabel(r'$v_{xe}/c$', fontsize=fontsize)
    ax.set_ylabel(r'$v_{ye}/c$', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax.set_title(title, fontsize=fontsize)
    ax.set_aspect('equal')

    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Particle count', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    fig.savefig(os.path.join(save_path, f"{label}.png"))
    plt.close(fig)


def animation(x, y, save_name, xlabel='', ylabel='', xmin=None, xmax=None,
              ymin=None, ymax=None, select='hist'):

    fig, ax = plt.subplots(constrained_layout=True)
    if xmin is None:
        xmin = np.nanmin(x)
    if xmax is None:
        xmax = np.nanmax(x)
    if ymin is None:
        ymin = np.nanmin(y)
    if ymax is None:
        ymax = np.nanmax(y)

    updates = make_updates(fig, ax, x, y, xmin, xmax, ymin, ymax, xlabel,
                           ylabel, save_name)

    frames = range(0, nt, dt_skip)
    ani = FuncAnimation(fig, updates[f"{select}"], frames=frames,
                        interval=interval)

    if save_name.endswith(".gif"):
        ani.save(save_name, writer=PillowWriter(fps=fps))
    else:
        ani.save(save_name, fps=fps)

    plt.close(fig)

    return ani


def make_updates(fig, ax, x, y, xmin, xmax, ymin, ymax, xlabel, ylabel,
                 save_name):
    # png_prefix = save_name.rsplit('.', 1)[0]

    def update_hist(it):
        ax.clear()
        valid = ~(np.isnan(x[it]))
        hist, edges = np.histogram(x[it][valid], bins=100, density=True)
        vc = 0.5*(edges[:-1] + edges[1:])
        ax.plot(vc, hist)
        ax.set_xlabel(f"{xlabel}", fontsize=fontsize)
        ax.set_ylabel(f"{ylabel}", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
        twpe = it * dt * wpe
        ax.set_title(f"$t\\omega_{{pe}} = {twpe:.2f}$", fontsize=fontsize)
        # nearest = round(twpe / 10) * 10
        # if abs(twpe - nearest) <= dt_skip * dt * wpe / 2:
        #     fig.savefig(
        #         f"{png_prefix}_{nearest:03.0f}.png",
        #         dpi=300
        #         )

    def update_raw(it):
        ax.clear()
        valid = ~(np.isnan(y[it]))
        ax.plot(x, y[it][valid])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(f"{xlabel}", fontsize=fontsize)
        ax.set_ylabel(f"{ylabel}", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
        twpe = it * dt * wpe
        ax.set_title(f"$t\\omega_{{pe}} = {twpe:.2f}$", fontsize=fontsize)
        # nearest = round(twpe / 10) * 10
        # if abs(twpe - nearest) <= dt_skip * dt * wpe / 2:
        #     fig.savefig(
        #         f"{png_prefix}_{nearest:03.0f}.png",
        #         dpi=300
        #         )

    def update_phase(it):
        ax.clear()
        valid = ~(np.isnan(x[it]) | np.isnan(y[it]))
        ax.scatter(x[it][valid], y[it][valid], s=s, color="blue")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(f"{xlabel}", fontsize=fontsize)
        ax.set_ylabel(f"{ylabel}", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
        twpe = it * dt * wpe
        ax.set_title(f"$t\\omega_{{pe}} = {twpe:.2f}$", fontsize=fontsize)
        # nearest = round(twpe / 10) * 10
        # if abs(twpe - nearest) <= dt_skip * dt * wpe / 2:
        #     fig.savefig(
        #         f"{png_prefix}_{nearest:03.0f}.png",
        #         dpi=300
        #         )

    return {
        "hist": update_hist,
        "raw": update_raw,
        "phase": update_phase,
    }


def save_animation(save, index, dx, save_fig_path):

    animation(index*dx, save["rhoe"], save_name=f"{save_fig_path}/rho_e.gif",
                xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$\\rho_e$',
                xmin=None, xmax=None, ymin=None, ymax=None,
                select='raw')
    animation(index*dx, save["rhoi"], save_name=f"{save_fig_path}/rho_i.gif",
                xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$\\rho_i$',
                xmin=None, xmax=None, ymin=None, ymax=None,
                select='raw')
    animation(index*dx, save["rhoei"], save_name=f"{save_fig_path}/rho.gif",
                xlabel='$x(*\\omega_{pe}/c)$', ylabel='$\\rho$',
                xmin=None, xmax=None, ymin=None, ymax=None,
                select='raw')
    animation(index*dx, save["ex"], save_name=f"{save_fig_path}/ex.gif",
                xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$E_{x}$',
                xmin=None, xmax=None, ymin=None, ymax=None,
                select='raw')
    animation(index*dx, save["phi"], save_name=f"{save_fig_path}/phi.gif",
                xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$\\phi$',
                xmin=None, xmax=None, ymin=None, ymax=None,
                select='raw')
    velocity_e = np.sqrt(save["vx"]**2 + save["vy"]**2 + save["vz"]**2)
    animation(velocity_e, save["vx"], save_name=f"{save_fig_path}/ve_f.gif",
                xlabel='$v_{e}(/c)$', ylabel='$f(v_{e})$',
                xmin=None, xmax=None, ymin=None, ymax=None,
                select='hist')
    velocity_i = np.sqrt(save["vxi"]**2 + save["vyi"]**2 + save["vzi"]**2)
    animation(velocity_i, save["vxi"], save_name=f"{save_fig_path}/vi_f.gif",
                xlabel='$v_{i}(/c)$', ylabel='$f(v_{i})$',
                xmin=None, xmax=None, ymin=None, ymax=None,
                select='hist')
    animation(save["vx"], save["vy"], save_name=f"{save_fig_path}/vx-vy.gif",
                xlabel='$v_{xe}(/c)$', ylabel='$v_{ye}(/c)$',
                xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
                select='phase')
    animation(save["x"]*dx, save["vx"], save_name=f"{save_fig_path}/x-vx.gif",
                xlabel='$x_e(*\\omega_{pe}/c)$', ylabel='$v_{xe}(/c)$',
                xmin=None, xmax=None, ymin=-0.5, ymax=0.5,
                select='phase')
    animation(save["vxi"], save["vyi"], save_name=f"{save_fig_path}/vxi-vyi.gif",
                xlabel='$v_{xi}(/c)$', ylabel='$v_{yi}(/c)$',
                xmin=-0.25, xmax=0.25, ymin=-0.25, ymax=0.25,
                select='phase')
    animation(save["xi"]*dx, save["vxi"], save_name=f"{save_fig_path}/xi-vxi.gif",
                xlabel='$x_i(*\\omega_{pe}/c)$', ylabel='$v_{xi}(/c)$',
                xmin=None, xmax=None, ymin=-0.25, ymax=0.25,
                select='phase')


def save_dispersion(save, index, dx, save_fig_path):

    if xmin == 0:
        xi = np.linspace(0, Tmax, 10000)
        eta = np.linspace(0, Tmax, 10000)
    else:
        xi = np.linspace(0.01, Tmax, 10000)
        eta = np.linspace(0.01, Tmax, 10000)
    w_xi, w_xi_m, w_eta, w_eta_m, xi_m, eta_m = \
            dispersion_relation(xmin, xmax, xi, eta)
    ez_m, bz_m = fluctuation_profile(xmin, xi_m, eta_m, index*dx)
    # dispersion_plot(save["ex"], save_fig_path, title=r'$E_x(k,\omega)$',
    #                 label='Ex_wk')
    # dispersion_plot(save["ey"], save_fig_path, title=r'$E_y(k,\omega)$',
    #                 label='Ey_wk')
    dispersion_plot(save["ez"], ez_m, xi_m, w_xi_m, save_fig_path, title=r'$E_z(T_m,\omega)$',
                    label='Ez_wk', flag=True)
    # dispersion_plot(save["by"], save_fig_path, title=r'$B_y(k,\omega)$',
    #                 label='By_wk')
    dispersion_plot(save["bz"], bz_m, eta_m, w_eta_m, save_fig_path, title=r'$B_z(T_m,\omega)$',
                    label='Bz_wk', flag=False)

    theory_plot(xi, w_xi, xi_m, w_xi_m, eta, w_eta, eta_m, w_eta_m, save_fig_path,
                    label='Dis.relation')


def save_phase_speed(vx, vy, vxi, vyi, save_fig_path):

    phase_speed(vx, vy, save_fig_path, title='Electron phase space',
                label='Electron_phase', vmin=None, vmax=None)
    phase_speed(vxi, vyi, save_fig_path, title='Ion phase space',
                label='Ion_phase', vmin=None, vmax=None)


def save_field_plot(save, save_fig_path):

    field_plot(save["ex"], save_fig_path, title=r'$E_x(x,t)$', label='Ex_xt')
    field_plot(save["ey"], save_fig_path, title=r'$E_y(x,t)$', label='Ey_xt')
    field_plot(save["ez"], save_fig_path, title=r'$E_z(x,t)$', label='Ez_xt')
    field_plot(save["by"], save_fig_path, title=r'$B_y(x,t)$', label='By_xt')
    field_plot(save["bz"], save_fig_path, title=r'$B_z(x,t)$', label='Bz_xt')
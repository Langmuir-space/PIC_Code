import numpy as np
import matplotlib.pyplot as plt
from params import dx, dt, xmax, wpe, dt_skip, fps, interval, fontsize, \
    s, bounds, bins, nt
import os
from matplotlib.animation import FuncAnimation, PillowWriter


def dispersion_plot(field, save_path, title, label):

    nt, nx = field.shape
    vmin = bounds[0]
    vmax = bounds[1]
    kmin = -2*np.pi*32/xmax
    kmax = 2*np.pi*32/xmax
    wmin = -2*np.pi*64/(nt*dt)
    wmax = 2*np.pi*64/(nt*dt)

    field_fft = np.fft.fftshift(np.fft.fft2(field))
    field_disp = np.log10(np.abs(field_fft) / nx / nt)

    k = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2*np.pi
    w = np.fft.fftshift(np.fft.fftfreq(nt, d=dt)) * 2*np.pi

    fig, ax = plt.subplots(constrained_layout=True)

    h = ax.imshow(field_disp, extent=[k.min(), k.max(), w.min(), w.max()],
                  origin='lower', cmap='jet', aspect='auto',
                  vmin=vmin, vmax=vmax)

    ax.set_xlim(kmin, kmax)
    ax.set_ylim(wmin, wmax)

    ax.set_xlabel(r'$k c/\omega_{pe}$', fontsize=fontsize)
    ax.set_ylabel(r'$\omega/\omega_{pe}$', fontsize=fontsize)
    ax.set_title(rf'{title}', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize - 2)
    ax.tick_params(axis='y', labelsize=fontsize - 2)

    cbar = fig.colorbar(h, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    fig.savefig(os.path.join(save_path, f"{label}.png"))
    plt.close(fig)


def field_plot(field, save_path, title, label):

    fig, ax = plt.subplots(constrained_layout=True)
    nt = field.shape[0]
    h = ax.imshow(field, extent=[0, xmax, 0, nt*dt], origin='lower',
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
    png_prefix = save_name.rsplit('.', 1)[0]

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

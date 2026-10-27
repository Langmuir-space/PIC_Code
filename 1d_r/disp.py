import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv
from scipy.optimize import brentq
from params import wpe, wpi, wce0, wci0


def dispersion_relation(Rmin, Rmax, xi, eta):

    def omega_xi(xi):

        w_xi = np.sqrt(wpe**2 + wpi**2 + xi**2)

        return w_xi

    def omega_eta(eta):

        c0 = 1
        c1 = 0
        c2 = - (eta**2 + wce0**2 + wci0**2 + 2*wpe**2 + 2*wpi**2)
        c3 = 0
        c4 = (wce0**2 + wci0**2 + wpe**2 + wpi**2)*eta**2 \
            + 2*(wpi**2*wce0**2 + wpe**2*wci0**2 + wpe**2*wpi**2) \
            + wpe**4 + wpi**4
        c5 = 0
        c6 = - ((wce0**2*wci0**2 + wci0**2*wpe**2 + wce0**2*wpi**2)*eta**2 \
                + (wpe**2*wci0 + wpi**2*wce0)**2)
        cof = [c0, c1, c2, c3, c4, c5, c6]
        w_eta = []
        for i in range(len(eta)):
            cof = [c0, c1, c2[i], c3, c4[i], c5, c6[i]]
            w_tmp = np.roots(cof)
            w_eta.append(w_tmp)
        w_eta = np.array(w_eta)

        return w_eta


    def boundary_condition(xi, eta, Rmin, Rmax):

        if Rmin == 0:
            def f_xi(xi):
                return jv(0, xi*Rmax)

            def f_eta(eta):
                return jv(1, eta*Rmax)

        elif Rmin > 0:
            def f_xi(xi):
                return jv(0, xi*Rmax)*yv(0, xi*Rmin) \
                    - jv(0, xi*Rmin)*yv(0, xi*Rmax)

            def f_eta(eta):
                return jv(1, eta*Rmax)*yv(1, eta*Rmin) \
                    - jv(1, eta*Rmin)*yv(1, eta*Rmax)

        xi_m = []
        eta_m = []

        for i in range(len(xi) - 1):
            if f_xi(xi[i]) * f_xi(xi[i+1]) < 0:
                root = brentq(f_xi, xi[i], xi[i+1])
                xi_m.append(root)

        for i in range(len(eta) - 1):
            if f_eta(eta[i]) * f_eta(eta[i+1]) < 0:
                root = brentq(f_eta, eta[i], eta[i+1])
                eta_m.append(root)

        return np.array(xi_m), np.array(eta_m)

    if Rmin < 0 or Rmax < 0:
        raise ValueError("Rmin and Rmax must be positive.")

    elif Rmin >= Rmax:
        raise ValueError("Rmin must be less than Rmax.")

    else:
        xi_m, eta_m = boundary_condition(xi, eta, Rmin, Rmax)
        w_xi = omega_xi(xi)
        w_xi_m = omega_xi(xi_m)
        w_eta = omega_eta(eta)
        w_eta_m = omega_eta(eta_m)

    return w_xi, w_xi_m, w_eta, w_eta_m, xi_m, eta_m


def fluctuation_profile(xmin, xi_m, eta_m, r):
    ez = []
    bz = []
    if xmin == 0:
        for i in range(len(xi_m)):
            ez.append(jv(0, xi_m[i]*r))
        for i in range(len(eta_m)):
            bz.append(jv(0, eta_m[i]*r))

    else:
        for i in range(len(xi_m)):
            ez.append(jv(0, xi_m[i]*r)
                      - jv(0, xi_m[i]*xmin)*yv(0, xi_m[i]*r)/yv(0, xi_m[i]*xmin))
        for i in range(len(eta_m)):
            bz.append(jv(0, eta_m[i]*r)
                      - jv(1, eta_m[i]*xmin)*yv(0, eta_m[i]*r)/yv(1, eta_m[i]*xmin))

    ez, bz = np.array(ez), np.array(bz)
    return ez, bz


def fluctuation_profile_ez(xmin, xi_m, r):
    ez = []
    if xmin == 0:
        for i in range(len(xi_m)):
            ez.append(jv(0, xi_m[i]*r))

    else:
        for i in range(len(xi_m)):
            ez.append(jv(0, xi_m[i]*r)
                      - jv(0, xi_m[i]*xmin)*yv(0, xi_m[i]*r)/yv(0, xi_m[i]*xmin))

    ez = np.array(ez)
    return ez


def fluctuation_profile_bz(xmin ,eta_m, r):
    bz = []
    if xmin == 0:
        for i in range(len(eta_m)):
            bz.append(jv(0, eta_m[i]*r))

    else:
        for i in range(len(eta_m)):
            bz.append(jv(0, eta_m[i]*r)
                      - jv(1, eta_m[i]*xmin)*yv(0, eta_m[i]*r)/yv(1, eta_m[i]*xmin))

    bz = np.array(bz)
    return bz


# xmin = 10
# xmax = 20

# Tmax = 5
# if xmin == 0:
#     xi = np.linspace(0, Tmax, 10000)
#     eta = np.linspace(0, Tmax, 10000)
# else:
#     xi = np.linspace(0.01, Tmax, 10000)
#     eta = np.linspace(0.01, Tmax, 10000)
# w_xi, w_xi_m, w_eta, w_eta_m, xi_m, eta_m = \
#         dispersion_relation(xmin, xmax, xi, eta)
# # ez_c, bz_c = fluctuation_profile(xmin, xi, eta, index*dx)
# index = np.linspace(xmin, xmax, 10000)
# ez_m, bz_m = fluctuation_profile(xmin, xi_m, eta_m, index)
# for i in range(5,6):
#     plt.plot(index, ez_m[i], linewidth=2.5, label=f'$T_{{m,{i+1}}}$')
#     plt.tick_params(labelsize=20)
#     plt.xlabel('$r(*\\omega_{pe}/c)$', fontsize=20)
#     # plt.ylabel('$J_0(\\xi R_{max}), J_1(\\eta R_{max})$', fontsize=15)
#     plt.title('$E_z(r)$', fontsize=20)
#     # plt.legend(fontsize=15)
#     # plt.legend(fontsize=15, loc='center left', bbox_to_anchor=(1.02, 0.5))
#     # plt.tight_layout(rect=[0, 0, 0.80, 1])
#     plt.tight_layout()
#     plt.savefig(rf'C:\Users\kasik\OneDrive - Kyushu University\myseminar\seminar_20261028\dispersion_relation8.png', dpi=300)
# plt.close()

# plt.scatter(xi, w_xi, s=0.1, c='black')
# plt.scatter(xi_m, w_xi_m, c='r', label='$T_m \,\ for \,\ TS$')
# plt.scatter(np.repeat(eta, w_eta.shape[1]), w_eta.real.ravel(),
#             s=0.1, c='black')
# plt.scatter(np.repeat(eta_m, w_eta_m.shape[1]), w_eta_m.real.ravel(),
#             c='blue', label='$T_m \,\ for \,\ TM$')
# plt.xlim(0, None)
# wmax = 5
# plt.ylim(0, wmax)
# plt.xlabel('$T (*c/\\omega_{pe})$', fontsize=20)
# plt.ylabel('$\\omega / \\omega_{pe}$', fontsize=20)
# plt.title(f'$\\omega_{{ce}}/\\omega_{{pe}}={wce0}, \
#         \\omega_{{ci}}/\\omega_{{pe}}={wci0}, \
#         {xmin} \\leq r \\leq {xmax}$', fontsize=18)
# plt.tick_params(labelsize=20)
# plt.legend(fontsize=15)
# plt.tight_layout()
# plt.savefig(rf'C:\Users\kasik\OneDrive - Kyushu University\myseminar\seminar_20261028\dispersion_relation6.png', dpi=300)
# plt.show()

# plt.plot(xi, jv(0, xi*xmax), linewidth=1.5, c='black')
# plt.plot(eta, jv(1, eta*xmax), linewidth=1.5, c='black')
# plt.plot(xi, jv(0, xi*xmax)*yv(0, xi*xmin) - jv(0, xi*xmin)*yv(0, xi*xmax), linewidth=1.5, c='black')
# plt.plot(eta, jv(1, xi*xmax)*yv(1, xi*xmin) - jv(1, xi*xmin)*yv(1, xi*xmax), linewidth=1.5, c='black')
# plt.scatter(xi_m, np.zeros_like(xi_m), color='red', zorder=5, label='$T_m \,\ for \,\ TS$')
# plt.scatter(eta_m, np.zeros_like(eta_m), color='blue', zorder=5, label='$T_m \,\ for \,\ TM$')
# plt.axhline(y=0, color='black', linewidth=1)
# plt.tick_params(labelsize=20)
# plt.xlabel('$T(*c/\\omega_{pe})$', fontsize=20)
# # plt.ylabel('$J_0(\\xi R_{max}), J_1(\\eta R_{max})$', fontsize=15)
# # plt.title('$J_0(T R_{max}), J_1(T R_{max})$', fontsize=20)
# plt.legend(fontsize=15)
# plt.tight_layout()
# plt.savefig(rf'C:\Users\kasik\OneDrive - Kyushu University\myseminar\seminar_20261028\dispersion_relation7.png', dpi=300)
# plt.show()


# plt.plot(xi, jv(0, xi*xmax)*yv(0, xi*xmin) - jv(0, xi*xmin)*yv(0, xi*xmax), linewidth=2.5)
# plt.scatter(xi_m, np.zeros_like(xi_m), color='red', zorder=5, label='$T_m$')
# plt.axhline(y=0, color='black', linewidth=1)
# plt.tick_params(labelsize=20)
# plt.xlabel('$T(*c/\\omega_{pe})$', fontsize=20)
# # plt.ylabel('$J_0(\\xi R_{max}), J_1(\\eta R_{max})$', fontsize=15)
# plt.title('$J_0(T R_{max})Y_0(T R_{min}) - Y_0(T R_{max})J_0(T R_{min})$', fontsize=20)
# plt.legend(fontsize=20)
# plt.tight_layout()
# plt.savefig(rf'C:\Users\kasik\OneDrive - Kyushu University\myseminar\seminar_20261028\dispersion_relation3.png', dpi=300)
# plt.show()


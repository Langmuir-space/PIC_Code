import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv
from scipy.optimize import brentq
from params import wpe, wpi, wce0, wci0, xmin, xmax

# xmax = 6
ximax = 6
etamax = 6
xi = np.linspace(0.001, ximax, 10000)
eta = np.linspace(0.001, etamax, 10000)


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

        xi_k = []
        eta_k = []

        for i in range(len(xi) - 1):
            if f_xi(xi[i]) * f_xi(xi[i+1]) < 0:
                root = brentq(f_xi, xi[i], xi[i+1])
                xi_k.append(root)

        for i in range(len(eta) - 1):
            if f_eta(eta[i]) * f_eta(eta[i+1]) < 0:
                root = brentq(f_eta, eta[i], eta[i+1])
                eta_k.append(root)

        return np.array(xi_k), np.array(eta_k)

    if Rmin < 0 or Rmax < 0:
        raise ValueError("Rmin and Rmax must be positive.")

    elif Rmin >= Rmax:
        raise ValueError("Rmin must be less than Rmax.")

    else:
        xi_k, eta_k = boundary_condition(xi, eta, Rmin, Rmax)
        w_xi = omega_xi(xi)
        w_xi_k = omega_xi(xi_k)
        w_eta = omega_eta(eta)
        w_eta_k = omega_eta(eta_k)

    return w_xi, w_xi_k, w_eta, w_eta_k, xi_k, eta_k


w_xi, w_xi_k, w_eta, w_eta_k, xi_k, eta_k = \
    dispersion_relation(xmin, xmax, xi, eta)
plt.scatter(xi, w_xi, s=0.1, c='black')
plt.scatter(xi_k, w_xi_k, c='r', label='$\\xi_k$')
plt.scatter(np.repeat(eta, w_eta.shape[1]), w_eta.real.ravel(),
            s=0.1, c='black')
plt.scatter(np.repeat(eta_k, w_eta_k.shape[1]), w_eta_k.real.ravel(),
            c='blue', label='$\\eta_k$')
plt.xlim(0, None)
plt.ylim(0, None)
plt.xlabel('$\\xi, \\eta \\,(*c/\\omega_{pe})$', fontsize=15)
plt.ylabel('$\\omega / \\omega_{pe}$', fontsize=15)
plt.title(f'$\\omega = \\omega(\\xi), \\omega = \
          \\omega(\\eta), {xmin} \\leq r \\leq {xmax}$', fontsize=15)
plt.tick_params(labelsize=15)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

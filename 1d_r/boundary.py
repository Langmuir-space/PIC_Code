from params import nx, x0


def ptcle_bc(x, vx, vy, vz):
    mask_in = x <= x0
    x[mask_in] = 2*x0 - x[mask_in]
    vx[mask_in] = -vx[mask_in]
    vy[mask_in] = vy[mask_in]
    vz[mask_in] = vz[mask_in]

    mask_out = x >= nx + x0
    x[mask_out] = 2*(nx + x0) - x[mask_out]
    vx[mask_out] = -vx[mask_out]
    vy[mask_out] = vy[mask_out]
    vz[mask_out] = vz[mask_out]

    return x, vx, vy, vz

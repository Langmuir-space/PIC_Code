import datetime
import os
import shutil
import numpy as np


def make_dic(base_save_path):
    dt_now = datetime.datetime.now()
    Y = dt_now.year
    M = dt_now.month
    D = dt_now.day
    mi = dt_now.minute
    S = dt_now.second
    H = int(dt_now.hour)
    file_name = ('{}{:02}{:02}{:02}{:02}{:02}'.format(Y, M, D, H, mi, S))
    save_path = os.path.join(base_save_path, file_name)
    os.makedirs(save_path, exist_ok=True)
    save_text_path = os.path.join(save_path, 'Text')
    os.makedirs(save_text_path, exist_ok=True)
    save_fig_path = os.path.join(save_path, 'Fig')
    os.makedirs(save_fig_path, exist_ok=True)
    save_code_path = os.path.join(save_path, 'Code')
    shutil.copytree('./', save_code_path)

    return save_text_path, save_fig_path


def tdma_pre(a, b, c):
    n = len(b)
    cp = np.zeros(n)
    bp = np.zeros(n)

    bp[0] = b[0]
    cp[0] = c[0] / bp[0]

    for i in range(1, n):
        bp[i] = b[i] - a[i] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / bp[i]
        else:
            cp[i] = 0.0

    return bp, cp


def tdma_solve(a, bp, cp, d):
    n = len(d)
    x = np.zeros_like(d)

    x[0] = d[0]

    for i in range(1, n):
        x[i] = d[i] - a[i] * x[i - 1] / bp[i - 1]

    x[-1] = x[-1]/bp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = x[i]/bp[i] - cp[i] * x[i + 1]

    return x

import datetime
import os
import shutil


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

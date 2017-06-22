import numpy as np


def apply_offset(t, mass, mid):
    i_mid = 0
    near = mid
    for i, it in enumerate(t):
        diff = abs(it - mid)
        if diff < near:
            i_mid = i
            near = diff

    offset = mass[i_mid]
    return mass - offset


def align_against(t: np.ndarray, mass: np.ndarray, t_ref: np.ndarray, mass_ref: np.ndarray) -> np.ndarray:
    t_start = np.min(t)
    i_align = 0
    min_err = np.inf

    for i, t in enumerate(t_ref):
        t_err = abs(t - t_start)
        if t_err < min_err:
            min_err = t_err
            i_align = i

    offset = mass_ref[i_align]
    return mass + offset

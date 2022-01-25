import numpy as np


def apply_offset(t, mass, mid):
    offset = np.interp(mid, t, mass)
    return mass - offset


def align_against(t: np.ndarray, mass: np.ndarray, t_ref: np.ndarray, mass_ref: np.ndarray) -> np.ndarray:
    offset = np.interp(t[0], t_ref, mass_ref) - mass[0]
    return mass + offset

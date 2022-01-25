from .combine import weighted_combine as ts_combine
from .functions import ts2m, match

import numpy as np


def sum_series(ts, data, ret_mask=False):
    if len(ts) == 1:
        return ts[0], data[0]

    t, _ = ts_combine(ts, data)
    out = np.zeros(t.shape, dtype=np.float64)

    beg_t = np.min(ts[0])
    end_t = np.max(ts[0])

    for i, times in enumerate(ts):
        tm, dm = ts2m(times, data[i])
        i1, i2 = match(t, tm, 1e-4)
        out[i1] += dm[i2]

        min_tm = np.min(tm)
        max_tm = np.max(tm)
        if min_tm > beg_t:
            beg_t = min_tm
        if max_tm < end_t:
            end_t = max_tm

    ok = np.logical_and(
        t >= beg_t,
        t <= end_t
    )
    if ret_mask:
        return t[ok], out[ok], ok
    return t[ok], out[ok]

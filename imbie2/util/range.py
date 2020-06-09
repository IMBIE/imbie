"""
calculate per-year range of a collection
"""
import numpy as np

from imbie2.model.collections import WorkingMassRateCollection
from imbie2.model.series import WorkingMassRateDataSeries


def calc_range(coll: WorkingMassRateCollection):
    s_year = np.floor(min([s.min_time for s in coll]))
    e_year = np.ceil(max([s.max_time for s in coll]))

    years = np.arange(s_year, e_year, 1, dtype=np.float) + .5
    max_ = np.empty_like(years) * np.nan
    min_ = np.empty_like(years) * np.nan

    for i, y in enumerate(years):
        for s in coll:
            if y < s.min_time:
                continue
            if y > s.max_time:
                continue

            val = np.interp(y, s.t, s.dmdt)

            if np.isnan(min_[i]) or min_[i] > val:
                min_[i] = val
            if np.isnan(max_[i]) or max_[i] < val:
                max_[i] = val
    
    range_ = max_ - min_
    return years, range_

def calc_sd(coll: WorkingMassRateCollection):
    s_year = np.floor(min([s.min_time for s in coll]))
    e_year = np.ceil(max([s.max_time for s in coll]))

    years = np.arange(s_year, e_year, 1, dtype=np.float) + .5
    vals = np.empty_like(years)

    for i, y in enumerate(years):
        v = []
        for s in coll:
            if y < s.min_time:
                continue
            if y > s.max_time:
                continue

            val = np.interp(y, s.t, s.dmdt)
            v.append(val)

        vals[i] = np.nanstd(v)

    return years, vals
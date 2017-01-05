from .data_series import DataSeries
import numpy as np
import math

from imbie2.util.functions import match
from imbie2.const.basins import BasinGroup


class MassRateDataSeries(DataSeries):

    @property
    def min_rate(self):
        ok = np.isfinite(self.dMdt)
        return np.min(self.dMdt[ok])

    @property
    def max_rate(self):
        ok = np.isfinite(self.dMdt)
        return np.max(self.dMdt[ok])

    def __init__(self, user, user_group, data_group, basin_group, basin_id,
                 basin_area, t_start, t_end, area, rate, errs, computed=False,
                 merged=False):
        super().__init__(
            user, user_group, data_group, basin_group, basin_id, basin_area,
            computed, merged
        )
        self.t0 = t_start
        self.t1 = t_end
        self.dMdt = rate
        self.dMdt_err = errs
        self.a = area

    def _set_min_time(self, min_t):
        ok = np.ones(self.t0.shape, dtype=bool)

        for i, t0 in enumerate(self.t0):
            if t0 < min_t:
                self.t0[i] = min_t
            if self.t1[i] < min_t:
                ok[i] = False

        self.t0 = self.t0[ok]
        self.t1 = self.t1[ok]
        self.dMdt = self.dMdt[ok]
        self.dMdt_err = self.dMdt_err[ok]
        self.a = self.a[ok]

    def _set_max_time(self, max_t):
        ok = np.ones(self.t0.shape, dtype=bool)

        for i, t1 in enumerate(self.t1):
            if t1 > max_t:
                self.t1[i] = max_t
            if self.t0[i] > max_t:
                ok[i] = False

        self.t0 = self.t0[ok]
        self.t1 = self.t1[ok]
        self.dMdt = self.dMdt[ok]
        self.dMdt_err = self.dMdt_err[ok]
        self.a = self.a[ok]

    def _get_min_time(self):
        return min(np.min(self.t1), np.min(self.t0))

    def _get_max_time(self):
        return max(np.max(self.t1), np.max(self.t0))

    @property
    def t(self):
        return (self.t0 + self.t1) / 2

    @classmethod
    def derive_rates(cls, mass_data):
        t0 = mass_data.t[:-1]
        t1 = mass_data.t[1:]
        dmdt = np.diff(mass_data.dM)
        area = (mass_data.a[:-1] + mass_data.a[1:]) / 2.

        return cls(
            mass_data.user, mass_data.user_group, mass_data.data_group, mass_data.basin_group,
            mass_data.basin_id, mass_data.basin_area, t0, t1, area, dmdt, mass_data.dM_err,
            computed=True
        )

    def __len__(self):
        return len(self.t0)

    @property
    def sigma(self):
        return math.sqrt(
            np.nanmean(np.square(self.dMdt_err))
        ) # / math.sqrt(len(self))

    @property
    def mean(self):
        return np.nanmean(self.dMdt)

    @classmethod
    def merge(cls, a, b):
        ia, ib = match(a.t0, b.t0)
        if a.user.lower() == "helm":
            print(a.t0, b.t0)
            print(a.t1, b.t1)
            print(ia, ib)

        if len(a) != len(b):
            return None
        if len(ia) != len(a) or len(ib) != len(b):
            return None

        t0 = a.t0[ia]
        t1 = a.t1[ia]
        m = (a.dMdt[ia] + b.dMdt[ib]) / 2.
        e = np.sqrt((np.square(a.dMdt_err[ia]) +
                     np.square(b.dMdt_err[ib])) / 2.)
        ar = (a.a[ia] + b.a[ib]) / 2.

        comp = a.computed or b.computed

        return cls(
            a.user, a.user_group, a.data_group, BasinGroup.sheets,
            a.basin_id, a.basin_area, t0, t1, ar, m, e, comp, merged=True
        )

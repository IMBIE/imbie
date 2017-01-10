from .data_series import DataSeries
import numpy as np

from imbie2.util.functions import ts2m, match
from imbie2.util.offset import apply_offset
from imbie2.const.basins import BasinGroup
import imbie2.model as model


class MassChangeDataSeries(DataSeries):

    @property
    def min_mass(self):
        ok = np.isfinite(self.dm)
        return np.min(self.dm[ok])

    @property
    def max_mass(self):
        ok = np.isfinite(self.dm)
        return np.max(self.dm[ok])

    def __init__(self, user, user_group, data_group, basin_group, basin_id,
                 basin_area, time, area, mass, errs, computed=False, merged=False):
        super().__init__(
            user, user_group, data_group, basin_group, basin_id, basin_area,
            computed, merged
        )
        self.t, self.dm = ts2m(time, mass)
        # _, self.a = ts2m(time, area)
        self.a = area
        _, self.errs = ts2m(time, errs)
        # self.t = time
        # self.mass = mass
        # self.errs = errs
        # self.a = area

    def _get_min_time(self):
        return np.min(self.t)

    def _get_max_time(self):
        return np.max(self.t)

    def _set_min_time(self, min_t):
        ok = self.t >= min_t

        self.t = self.t[ok]
        self.dm = self.dm[ok]
        self.a = self.a[ok]
        self.errs = self.errs[ok]

    def _set_max_time(self, max_t):
        ok = self.t <= max_t

        self.t = self.t[ok]
        self.dm = self.dm[ok]
        self.a = self.a[ok]
        self.errs = self.errs[ok]

    @classmethod
    def accumulate_mass(cls, rate_data, offset=None):
        if isinstance(rate_data, model.series.MassRateDataSeries):
            t = (rate_data.t0 + rate_data.t1) / 2.
            dM = np.cumsum(rate_data.dmdt) / 12.
            err = np.cumsum(rate_data.errs) / 12.

        elif isinstance(rate_data, model.series.WorkingMassRateDataSeries):
            t = rate_data.t             # t = (rate_data.t[:-1] + rate_data.t[1:]) / 2.
            dM = np.cumsum(rate_data.dmdt) / 12.
            err = np.cumsum(rate_data.errs) / 12.

        else: raise TypeError("Rates data expected")

        if offset is not None:
            dM = apply_offset(t, dM, offset)

        return cls(
            rate_data.user, rate_data.user_group, rate_data.data_group, rate_data.basin_group,
            rate_data.basin_id, rate_data.basin_area, t, rate_data.a, dM, err, computed=True
        )

    def differentiate(self):
        return model.series.MassRateDataSeries.derive_rates(self).chunk_rates()

    def __len__(self):
        return len(self.t)

    @classmethod
    def merge(cls, a, b):
        ia, ib = match(a.t, b.t)
        if len(a) != len(b):
            return None
        if len(ia) != len(a) or len(ib) != len(b):
            return None
        if len(ia) == 0:
            return None

        t = a.t[ia]
        m = (a.dm[ia] + b.dm[ib]) / 2.
        e = np.sqrt((np.square(a.errs[ia]) +
                     np.square(b.errs[ib])) / 2.)
        ar = (a.a[ia] + b.a[ib]) / 2.

        comp = a.computed or b.computed

        return cls(
            a.user, a.user_group, a.data_group, BasinGroup.sheets,
            a.basin_id, a.basin_area, t, ar, m, e, comp, merged=True
        )

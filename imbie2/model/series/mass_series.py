from .data_series import DataSeries
import numpy as np

from imbie2.util.functions import ts2m, match
from imbie2.util.offset import apply_offset
from imbie2.const.basins import BasinGroup, Basin
import imbie2.model as model
from typing import Optional


class MassChangeDataSeries(DataSeries):

    @property
    def min_mass(self) -> float:
        ok = np.isfinite(self.mass)
        return np.min(self.mass[ok])

    @property
    def max_mass(self) -> float:
        ok = np.isfinite(self.mass)
        return np.max(self.mass[ok])

    def __init__(self, user: Optional[str], user_group: Optional[str], data_group: Optional[str],
                 basin_group: BasinGroup, basin_id: Basin, basin_area: float, time: np.ndarray, area: np.ndarray,
                 mass: np.ndarray, errs: np.ndarray, computed: bool=False, merged: bool=False, aggregated: bool=False,
                 contributions: int=1):
        super().__init__(
            user, user_group, data_group, basin_group, basin_id, basin_area,
            computed, merged, aggregated, contributions
        )
        self.t, self.mass = ts2m(time, mass)
        # _, self.a = ts2m(time, area)
        self.a = area
        _, self.errs = ts2m(time, errs)
        # self.t = time
        # self.mass = mass
        # self.errs = errs
        # self.a = area

    def _get_min_time(self) -> float:
        return np.min(self.t)

    def _get_max_time(self) -> float:
        return np.max(self.t)

    def _set_min_time(self, min_t: float) -> None:
        ok = self.t >= min_t

        self.t = self.t[ok]
        self.mass = self.mass[ok]
        self.a = self.a[ok]
        self.errs = self.errs[ok]

    def _set_max_time(self, max_t: float) -> None:
        ok = self.t <= max_t

        self.t = self.t[ok]
        self.mass = self.mass[ok]
        self.a = self.a[ok]
        self.errs = self.errs[ok]

    @classmethod
    def accumulate_mass(cls, rate_data, offset: float=None) -> "MassChangeDataSeries":
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
            rate_data.basin_id, rate_data.basin_area, t, rate_data.a, dM, err, computed=True,
            aggregated=rate_data.aggregated
        )

    def differentiate(self) -> "model.series.MassRateDataSeries":
        return model.series.MassRateDataSeries.derive_rates(self).chunk_rates()

    def __len__(self) -> int:
        return len(self.t)

    @classmethod
    def merge(cls, a: "MassChangeDataSeries", b: "MassChangeDataSeries") -> "MassChangeDataSeries":
        ia, ib = match(a.t, b.t)
        if len(a) != len(b):
            return None
        if len(ia) != len(a) or len(ib) != len(b):
            return None
        if len(ia) == 0:
            return None

        t = a.t[ia]
        m = (a.mass[ia] + b.mass[ib]) / 2.
        e = np.sqrt((np.square(a.errs[ia]) +
                     np.square(b.errs[ib])) / 2.)
        ar = (a.a[ia] + b.a[ib]) / 2.

        comp = a.computed or b.computed
        aggr = a.aggregated or b.aggregated

        return cls(
            a.user, a.user_group, a.data_group, BasinGroup.sheets,
            a.basin_id, a.basin_area, t, ar, m, e, comp, merged=True, aggregated=True
        )

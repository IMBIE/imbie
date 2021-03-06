from .data_series import DataSeries
import numpy as np

from imbie2.util.functions import ts2m, match, smooth_imbie
from imbie2.util.offset import apply_offset, align_against
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
                 contributions: int=1, interpolate: bool=True):
        super().__init__(
            user, user_group, data_group, basin_group, basin_id, basin_area,
            computed, merged, aggregated, contributions
        )
        if interpolate:
            self.t, self.mass = ts2m(time, mass)
            _, self.errs = ts2m(time, errs)
        else:
            self.t = time
            self.mass = mass
            self.errs = errs
            
        self.a = area

    def _get_min_time(self) -> float:
        return np.min(self.t)

    def _get_max_time(self) -> float:
        return np.max(self.t)

    def _set_min_time(self, min_t: float) -> None:
        ok = self.t >= min_t

        self.t = self.t[ok]
        self.mass = self.mass[ok]
        # cropping the area has been removed as it was causing indexing
        # errors. This isn't a currently a problem as the area isn't used
        # in the analysis. TODO: find a proper solution to this
            # self.a = self.a[ok]
        self.errs = self.errs[ok]

    def _set_max_time(self, max_t: float) -> None:
        ok = self.t <= max_t

        self.t = self.t[ok]
        self.mass = self.mass[ok]
        # self.a = self.a[ok[:len(self.a)]]
        self.errs = self.errs[ok]

    def reduce(self, interval: float=1., centre=None):
        mean_diff = np.mean(np.diff(self.t))
        if mean_diff >= interval:
            return self

        breaks = np.hstack(([0], np.argwhere(np.isnan(self.mass)).flat, [-1]))
        all_windows_mass = []
        all_windows_errs = []
        all_windows_t = []

        for start, final in zip(breaks[:-1], breaks[1:]):
            if start + 1 == final:
                continue

            is_last = (final == -1)

            min_t = self.t[start]
            max_t = self.t[final]
            half_i = interval / 2.

            if centre is None:
                t_new = np.arange(min_t+half_i, max_t-half_i, interval)
                t_new = np.hstack((t_new, [max_t-half_i]))
            else:
                t_new = np.arange(
                    np.floor(min_t) - centre, np.ceil(max_t) + centre, interval
                )
                t_new = t_new[(t_new > min_t) & (t_new < max_t)]

            window_mass = np.interp(t_new, self.t, self.mass)
            window_errs = np.interp(t_new, self.t, self.errs)
            window_t = t_new

            if not is_last:
                # add extra NaN record to create break
                window_mass = np.hstack((window_mass, [np.nan]))
                window_errs = np.hstack((window_errs, [np.nan]))
                window_t = np.hstack((window_t, [max_t+half_i]))
                
            all_windows_mass.append(window_mass)
            all_windows_errs.append(window_errs)
            all_windows_t.append(window_t)

        mass = np.hstack(all_windows_mass)
        errs = np.hstack(all_windows_errs)
        t_new = np.hstack(all_windows_t)

        a = np.empty_like(t_new) * np.nan

        return MassChangeDataSeries(
            self.user, self.user_group, self.data_group, self.basin_group, self.basin_id, self.basin_area,
            t_new, a, mass, errs, self.computed, self.merged, self.aggregated, self.contributions, interpolate=False
        )

    @classmethod
    def accumulate_mass(cls, rate_data, offset: float=None, ref_series: "MassChangeDataSeries"=None)\
            -> "MassChangeDataSeries":
        if isinstance(rate_data, model.series.MassRateDataSeries):
            t = (rate_data.t0 + rate_data.t1) / 2.
            dM = np.cumsum(rate_data.dmdt) / 12.
            err = np.cumsum(rate_data.errs) / 12.
            # TODO: confirm this
            n = t - t[0]; n[0] = 1.
            err /= np.sqrt(n)

        elif isinstance(rate_data, model.series.WorkingMassRateDataSeries):
            t = rate_data.t             # t = (rate_data.t[:-1] + rate_data.t[1:]) / 2.
            dM = np.cumsum(rate_data.dmdt) / 12.
            # err = np.sqrt(np.cumsum(np.square(rate_data.errs / 12)) / np.arange(1, rate_data.errs.size+1))
            err = np.sqrt(np.cumsum(np.square(rate_data.errs))) / np.sqrt(12) # FIXME: reset

        else: raise TypeError("Rates data expected")

        if offset is not None:
            assert ref_series is None

            dM = apply_offset(t, dM, offset)
        elif ref_series is not None:
            dM = align_against(t, dM, ref_series.t, ref_series.mass)

        return cls(
            rate_data.user, rate_data.user_group, rate_data.data_group, rate_data.basin_group,
            rate_data.basin_id, rate_data.basin_area, t, rate_data.a, dM, err, computed=True,
            aggregated=rate_data.aggregated
        )

    def differentiate(self) -> "model.series.MassRateDataSeries":
        return model.series.MassRateDataSeries.derive_rates(self).chunk_rates()

    def align(self, reference: "MassChangeDataSeries") -> "MassChangeDataSeries":
        mass = align_against(self.t, self.mass, reference.t, reference.mass)

        return MassChangeDataSeries(
            self.user, self.user_group, self.data_group, self.basin_group, self.basin_id, self.basin_area,
            self.t, self.a, mass, self.errs, self.computed, self.merged, self.aggregated, self.contributions
        )

    def __len__(self) -> int:
        return len(self.t)

    def __bool__(self) -> bool:
        return len(self) > 0

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

    def smooth(self, window: float=13./12) -> "MassChangeDataSeries":
        if window is None:
            return self

        mass = smooth_imbie(self.t, self.mass, window)
        mass_err = self.errs.copy()

        return self.__class__(
            self.user, self.user_group, self.data_group, self.basin_group,
            self.basin_id, self.basin_area, self.t, self.a, mass, mass_err,
            self.computed, self.merged, self.aggregated, self.contributions
        )

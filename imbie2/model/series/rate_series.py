from .data_series import DataSeries
import numpy as np
import math

from imbie2.util.functions import match, smooth_imbie
from imbie2.util.combine import weighted_combine as ts_combine
from imbie2.util import dm_to_dmdt
from imbie2.const.basins import BasinGroup, Basin
from imbie2.const.error_methods import ErrorMethod
from imbie2.const.lsq_methods import LSQMethod
import imbie2.model as model

from typing import Optional, Tuple


class MassRateDataSeries(DataSeries):

    @property
    def min_rate(self) -> None:
        ok = np.isfinite(self.dmdt)
        return np.min(self.dmdt[ok])

    @property
    def max_rate(self) -> None:
        ok = np.isfinite(self.dmdt)
        return np.max(self.dmdt[ok])

    def __init__(self, user: Optional[str], user_group: Optional[str], data_group: Optional[str],
                 basin_group: BasinGroup, basin_id: Basin, basin_area: float, t_start: np.ndarray, t_end: np.ndarray,
                 area: np.ndarray, rate: np.ndarray, errs: np.ndarray, computed: bool=False, merged: bool=False,
                 aggregated: bool=False, contributions: int=1):
        super().__init__(
            user, user_group, data_group, basin_group, basin_id, basin_area,
            computed, merged, aggregated, contributions
        )
        self.t0 = t_start
        self.t1 = t_end
        self.dmdt = rate
        self.errs = errs
        self.a = area

    def _set_min_time(self, min_t: float) -> None:
        ok = np.ones(self.t0.shape, dtype=bool)

        for i, t0 in enumerate(self.t0):
            if t0 < min_t:
                self.t0[i] = min_t
            if self.t1[i] < min_t:
                ok[i] = False

        self.t0 = self.t0[ok]
        self.t1 = self.t1[ok]
        self.dmdt = self.dmdt[ok]
        self.errs = self.errs[ok]
        # self.a = self.a[ok]

    def _set_max_time(self, max_t: float) -> None:
        ok = np.ones(self.t0.shape, dtype=bool)

        for i, t1 in enumerate(self.t1):
            if t1 > max_t:
                self.t1[i] = max_t
            if self.t0[i] > max_t:
                ok[i] = False

        self.t0 = self.t0[ok]
        self.t1 = self.t1[ok]
        self.dmdt = self.dmdt[ok]
        self.errs = self.errs[ok]
        # self.a = self.a[ok]

    def _get_min_time(self) -> float:
        return min(np.min(self.t1), np.min(self.t0))

    def _get_max_time(self) -> float:
        return max(np.max(self.t1), np.max(self.t0))

    @property
    def t(self) -> np.ndarray:
        return (self.t0 + self.t1) / 2

    @classmethod
    def derive_rates(cls, mass_data: "model.series.MassChangeDataSeries") -> "MassRateDataSeries":
        t0 = mass_data.t[:-1]
        t1 = mass_data.t[1:]
        dmdt = np.diff(mass_data.mass)
        if mass_data.a is not None:
            area = (mass_data.a[:-1] + mass_data.a[1:]) / 2.
        else:
            area = None

        return cls(
            mass_data.user, mass_data.user_group, mass_data.data_group, mass_data.basin_group,
            mass_data.basin_id, mass_data.basin_area, t0, t1, area, dmdt, mass_data.errs,
            computed=True, aggregated=mass_data.aggregated
        )

    def __len__(self) -> int:
        return len(self.t0)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def sigma(self) -> float:
        return math.sqrt(
            np.nanmean(np.square(self.errs))
        )  # / math.sqrt(len(self))

    @property
    def mean(self) -> float:
        return np.nanmean(self.dmdt)

    @classmethod
    def merge(cls, a: "MassRateDataSeries", b: "MassRateDataSeries") -> "MassRateDataSeries":
        ia, ib = match(a.t0, b.t0)

        if len(a) != len(b):
            return None
        if len(ia) != len(a) or len(ib) != len(b):
            return None

        t0 = a.t0[ia]
        t1 = a.t1[ia]
        m = (a.dmdt[ia] + b.dmdt[ib]) / 2.
        e = np.sqrt((np.square(a.errs[ia]) +
                     np.square(b.errs[ib])) / 2.)
        ar = (a.a[ia] + b.a[ib]) / 2.

        comp = a.computed or b.computed
        aggr = a.aggregated or b.aggregated

        return cls(
            a.user, a.user_group, a.data_group, BasinGroup.sheets,
            a.basin_id, a.basin_area, t0, t1, ar, m, e, comp, merged=True, aggregated=aggr
        )

    def chunk_rates(self) -> "WorkingMassRateDataSeries":
        ok = self.t0 == self.t1

        time_chunks = [self.t0[ok]]
        dmdt_chunks = [self.dmdt[ok]]
        errs_chunks = [self.errs[ok]]

        if np.all(ok):
            t = time_chunks[0]
            dmdt = dmdt_chunks[0]
            errs = errs_chunks[0]
        else:
            for i in range(len(self)):
                if ok[i]: continue

                time_chunks.append(
                    np.asarray([self.t0[i], self.t1[i]])
                )
                dmdt_chunks.append(
                    np.asarray([self.dmdt[i], self.dmdt[i]])
                )
                errs_chunks.append(
                    np.asarray([self.errs[i], self.errs[i]])
                )
            t, dmdt = ts_combine(time_chunks, dmdt_chunks)
            _, errs = ts_combine(time_chunks, errs_chunks, error_method=ErrorMethod.rms)

        return WorkingMassRateDataSeries(
            self.user, self.user_group, self.data_group, self.basin_group, self.basin_id, self.basin_area,
            t, self.a, dmdt, errs, aggregated=self.aggregated
        )

    def integrate(self, offset: float=None) -> "model.series.MassChangeDataSeries":
        return model.series.MassChangeDataSeries.accumulate_mass(self, offset=offset)


class WorkingMassRateDataSeries(DataSeries):
    def __init__(self, user: Optional[str], user_group: Optional[str], data_group: Optional[str],
                 basin_group: BasinGroup, basin_id: Basin, basin_area: float, time: np.ndarray, area: np.ndarray,
                 dmdt: np.ndarray, errs: np.ndarray, computed: bool=False, merged: bool=False, aggregated: bool=False,
                 contributions: int=1, truncate: Tuple[float, float]=None):
        super().__init__(
            user, user_group, data_group, basin_group, basin_id, basin_area,
            computed, merged, aggregated, contributions
        )
        self.t = time
        self.a = area
        self.dmdt = dmdt
        self.errs = errs
        self.trunc_extent = truncate

    def get_truncated(self) -> "WorkingMassRateDataSeries":
        if self.trunc_extent is None:
            return self
        start, end = self.trunc_extent
        return self.truncate(start, end)

    @property
    def min_rate(self) -> float:
        ok = np.isfinite(self.dmdt)
        return np.min(self.dmdt[ok])

    @property
    def max_rate(self) -> float:
        ok = np.isfinite(self.dmdt)
        return np.max(self.dmdt[ok])

    @property
    def sigma(self) -> float:
        return math.sqrt(
            np.nanmean(np.square(self.errs))
        )  # / math.sqrt(len(self))

    @property
    def mean(self) -> float:
        return np.nanmean(self.dmdt)

    def _get_min_time(self) -> float:
        return np.min(self.t)

    def _get_max_time(self) -> float:
        return np.max(self.t)

    def _set_min_time(self, min_t: float, interp: bool=True) -> None:
        if min_t < self.t.min():
            return

        ok = self.t >= min_t

        if interp:
            # interpolate values @ new minimum
            new_dmdt = np.interp(min_t, self.t, self.dmdt)
            new_errs = np.interp(min_t, self.t, self.errs)
            
            self.t =\
                np.hstack((min_t, self.t[ok]))
            self.dmdt =\
                np.hstack((new_dmdt, self.dmdt[ok]))
            self.errs =\
                np.hstack((new_errs, self.errs[ok]))
        else:
            self.t = self.t[ok]
            self.dmdt = self.dmdt[ok]
            self.errs = self.errs[ok]
        # if self.a is not None:
        #     self.a = self.a[ok]
        

    def _set_max_time(self, max_t: float, interp: bool=True) -> None:
        if max_t > self.t.max():
            return

        ok = self.t < max_t

        if interp:
            new_dmdt = np.interp(max_t, self.t, self.dmdt)
            new_errs = np.interp(max_t, self.t, self.errs)
            
            self.t =\
                np.hstack((self.t[ok], max_t))
            self.dmdt =\
                np.hstack((self.dmdt[ok], new_dmdt))
            self.errs =\
                np.hstack((self.errs[ok], new_errs))
        else:
            self.t = self.t[ok]
            self.dmdt = self.dmdt[ok]
            self.errs = self.errs[ok]

    def integrate(self, offset: float=None, align: "MassChangeDataSeries"=None) -> "model.series.MassChangeDataSeries":
        return model.series.MassChangeDataSeries.accumulate_mass(self, offset=offset, ref_series=align)

    def reduce(self, interval: float=1., centre=None, backfill: bool=False, interp: bool=False) -> "WorkingMassRateDataSeries":

        mean_diff = np.mean(np.diff(self.t))
        if mean_diff >= interval:
            half_i = interval / 2.
            t_new = np.arange(self.t.min()+half_i, self.t.max()-half_i, interval)
            dmdt_interp = interp1d(self.t, self.dmdt, kind='nearest', fill_value="extrapolate")
            errs_interp = interp1d(self.t, self.errs, kind='nearest', fill_value="extrapolate")
            dmdt = dmdt_interp(t_new)
            errs = errs_interp(t_new)

            return WorkingMassRateDataSeries(
                self.user, self.user_group, self.data_group, self.basin_group, self.basin_id, self.basin_area,
                t_new, self.a, dmdt, errs, aggregated=self.aggregated, computed=self.computed, truncate=self.trunc_extent
            )

        breaks = np.hstack(([0], np.argwhere(np.isnan(self.dmdt)).flat, [-1]))
        all_windows_dmdt = []
        all_windows_errs = []
        all_windows_t = []

        for start, final in zip(breaks[:-1], breaks[1:]):
            if start + 1 == final:
                continue
            is_last = (final == -1)

            min_t = self.t[start]
            max_t = self.t[final]

            # min_t = self.t.min()
            # max_t = self.t.max()
            half_i = interval / 2.

            if centre is None:
                t_new = np.arange(min_t+half_i, max_t-half_i, interval)
                t_new = np.hstack((t_new, [max_t-half_i]))
            else:
                t_new = np.arange(
                    np.ceil(min_t) + centre, np.floor(max_t), interval
                )

            if interp:
                window_dmdt = np.interp(t_new, self.t, self.dmdt)
                window_errs = np.interp(t_new, self.t, self.errs)
                window_t = t_new
            else:
                n_points_out = len(t_new)
                n_max_window = len(self)

                window_dmdt = np.empty((n_points_out, n_max_window))
                window_errs = np.empty((n_points_out, n_max_window))

                for i, t in enumerate(t_new):
                    w_start = t-half_i
                    w_final = t+half_i

                    w_mask = np.logical_and(w_start <= self.t, self.t < w_final)
                    window_dmdt[i, :] = np.where(w_mask, self.dmdt, np.nan)
                    window_errs[i, :] = np.where(w_mask, self.errs, np.nan)

                window_dmdt = np.nanmean(window_dmdt, axis=1)
                window_errs = np.nanmean(window_errs, axis=1)
                window_t = t_new

            if backfill:
                t_backfill = np.arange(
                    min_t, max_t, 1./12.
                )
                t_backfill = np.hstack((t_backfill, [max_t]))

                dmdt_interp = interp1d(window_t, window_dmdt, kind='nearest', fill_value="extrapolate")
                errs_interp = interp1d(window_t, window_errs, kind='nearest', fill_value="extrapolate")
                window_dmdt = dmdt_interp(t_backfill)
                window_errs = errs_interp(t_backfill)
                window_t = t_backfill

            if not is_last:
                # add extra NaN record to create break
                window_dmdt = np.hstack((window_dmdt, [np.nan]))
                window_errs = np.hstack((window_errs, [np.nan]))
                window_t = np.hstack((window_t, [max_t+half_i]))
                
            all_windows_dmdt.append(window_dmdt)
            all_windows_errs.append(window_errs)
            all_windows_t.append(window_t)

        dmdt = np.hstack(all_windows_dmdt)
        errs = np.hstack(all_windows_errs)
        t_new = np.hstack(all_windows_t)

        return WorkingMassRateDataSeries(
            self.user, self.user_group, self.data_group, self.basin_group, self.basin_id, self.basin_area,
            t_new, self.a, dmdt, errs, aggregated=self.aggregated, computed=self.computed, truncate=self.trunc_extent
        )

    @classmethod
    def merge(cls, a: "WorkingMassRateDataSeries", b: "WorkingMassRateDataSeries") -> "WorkingMassRateDataSeries":
        ia, ib = match(a.t, b.t)

        if len(a) != len(b):
            return None
        if len(ia) != len(a) or len(ib) != len(b):
            return None

        t = a.t[ia]
        m = (a.dmdt[ia] + b.dmdt[ib]) / 2.
        e = np.sqrt((np.square(a.errs[ia]) +
                     np.square(b.errs[ib])) / 2.)
        # ar = (a.a[ia] + b.a[ib]) / 2.
        ar = None

        comp = a.computed or b.computed
        aggr = a.aggregated or b.aggregated

        return cls(
            a.user, a.user_group, a.data_group, BasinGroup.sheets,
            a.basin_id, a.basin_area, t, ar, m, e, comp, merged=True, aggregated=aggr
        )

    @classmethod
    def from_dm(cls, mass_series: "model.series.MassChangeDataSeries", truncate: bool=True, window: float=1.,
                method: LSQMethod=LSQMethod.normal) -> "WorkingMassRateDataSeries":
        dmdt, dmdt_err = dm_to_dmdt(
            mass_series.t, mass_series.mass, mass_series.errs,
            window, truncate=truncate, lsq_method=method
        )
        if truncate:
            finite_indicies = np.flatnonzero(
                np.isfinite(dmdt)
            )

            first_valid = finite_indicies.min()
            final_valid = finite_indicies.max()

            dmdt[:first_valid] = dmdt[first_valid]
            dmdt[final_valid:] = dmdt[final_valid]

            dmdt_err[:first_valid] = dmdt_err[first_valid]
            dmdt_err[final_valid:] = dmdt_err[final_valid]

            crop_to = mass_series.t[first_valid], mass_series.t[final_valid]
        else:
            crop_to = None

        return cls(
            mass_series.user, mass_series.user_group, mass_series.data_group, mass_series.basin_group,
            mass_series.basin_id, mass_series.basin_area, mass_series.t, mass_series.a, dmdt, dmdt_err,
            computed=True, truncate=crop_to
        )

    def smooth(self, window: float=13./12, clip=False, iters=1) -> "WorkingMassRateDataSeries":
        if window is None:
            return self

        dmdt = smooth_imbie(self.t, self.dmdt, window, iters)
        dmdt_err = smooth_imbie(self.t, self.errs, window, iters)

        if clip:
            margin = window / 2.

            first_valid = np.argwhere(self.t < self.t.min() + margin).max()
            final_valid = np.argwhere(self.t > self.t.max() - margin).min()

            dmdt[:first_valid] = dmdt[first_valid]
            dmdt[final_valid:] = dmdt[final_valid]

            dmdt_err[:first_valid] = dmdt_err[first_valid]
            dmdt_err[final_valid:] = dmdt_err[final_valid]

            crop_to = self.t[first_valid], self.t[final_valid]
        else:
            crop_to = None

        return WorkingMassRateDataSeries(
            self.user, self.user_group, self.data_group, self.basin_group,
            self.basin_id, self.basin_area, self.t, self.a, dmdt, dmdt_err,
            self.computed, self.merged, self.aggregated, truncate=crop_to
        )

    def __len__(self) -> int:
        return len(self.t)

    def chunk_rates(self):
        return self

    def truncate(self, min_time: float, max_time: float, interp: bool=True) -> "WorkingMassRateDataSeries":
        trunc = self.__class__(
            self.user, self.user_group, self.data_group, self.basin_group,
            self.basin_id, self.basin_area, self.t, self.a, self.dmdt, self.errs,
            self.computed, self.merged, self.aggregated
        )
        trunc.limit_times(min_time, max_time, interp=interp)

        if len(trunc.t) == 0:
            print(trunc.user, trunc.basin_id.value)
            print(min_time, max_time, self.t)
        return trunc

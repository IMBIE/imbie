from .collection import Collection
from imbie2.const.error_methods import ErrorMethod
from imbie2.const.average_methods import AverageMethod
from imbie2.model.series import WorkingMassRateDataSeries, MassRateDataSeries
from imbie2.util.combine import weighted_combine as ts_combine
from imbie2.util.sum_series import sum_series
from imbie2.util.functions import match
import imbie2.model as model

from typing import Iterator
import numpy as np


class MassRateCollection(Collection):
    def average(self, mode=None):
        raise NotImplementedError("method not available for raw dm/dt data (chunk this collection first)")

    def sum(self, error_method=ErrorMethod.sum):
        raise NotImplementedError("method not available for raw dm/dt data (chunk this collection first)")

    def __iter__(self) -> Iterator[MassRateDataSeries]:
        return super().__iter__()

    def chunk_series(self) -> "WorkingMassRateCollection":
        out = WorkingMassRateCollection()
        for series in self:
            chunked = series.chunk_rates()
            out.add_series(chunked)
        return out

    def integrate(self) -> "model.collections.MassChangeCollection":
        out = model.collections.MassChangeCollection()
        for series in self:
            out.add_series(series.integrate())

        return out

    def filter(self, **kwargs) -> "MassRateCollection":
        return super().filter(**kwargs)

    def __add__(self, other: "MassRateCollection") -> "MassRateCollection":
        return MassRateCollection(*(self.series + other.series))

    def __iadd__(self, other: "MassRateCollection") -> "MassRateCollection":
        self.series += other.series
        return self


class WorkingMassRateCollection(Collection):
    def __iter__(self) -> Iterator[WorkingMassRateDataSeries]:
        return super().__iter__()

    def average(self, mode: AverageMethod=AverageMethod.equal_groups, nsigma: float=None,
                error_mode: ErrorMethod=ErrorMethod.rss, export_data: str=None) -> WorkingMassRateDataSeries:
        if not self.series:
            return None
        elif len(self.series) == 1:
            return self.series[0]

        b_id = self.series[0].basin_id
        b_gp = self.series[0].basin_group
        b_a = self.series[0].basin_area

        u_gp = self.series[0].user_group
        d_gp = self.series[0].data_group

        for series in self:
            if series.basin_id != b_id:
                b_id = None
            if series.basin_group != b_gp:
                b_gp = None
            if series.basin_area != b_a:
                b_a = None
            if series.user_group != u_gp:
                u_gp = None
            if series.data_group != d_gp:
                d_gp = None

        ts = [series.t for series in self]
        ms = [series.dmdt for series in self]
        es = [series.errs for series in self]
        count = sum([series.contributions for series in self])

        if mode == AverageMethod.equal_groups or mode == AverageMethod.imbie1_compat:
            w = None
        elif mode == AverageMethod.equal_series:
            _max = max([s.contributions for s in self])
            w = [series.contributions/_max for series in self]
        elif mode == AverageMethod.inverse_errs:
            w = [np.reciprocal(e) for e in es]
        else:
            raise ValueError("Unknown averaging mode: \"{}\"".format(mode))

        t, m = ts_combine(ts, ms, w=w, nsigma=nsigma)

        if error_mode in [ErrorMethod.rms_deviation, ErrorMethod.rss_dev_epoch, ErrorMethod.constant_dev, ErrorMethod.max_error]:
            # calculate RMS of difference between contributions
            #  and mean at each epoch
            devs = np.empty((len(self), t.size)) * np.nan

            for i, s in enumerate(self):
                i_m, i_s = match(t, s.t, 1./48)
                devs[i, i_m] = s.dmdt[i_s] - m[i_m]

            e = np.sqrt(np.nanmean(np.square(devs), axis=0))

            all_members = np.isfinite(np.product(devs, axis=0)) # find epochs w/o nan in 'devs'
            const_e = np.ones_like(e) * np.mean(e[all_members]) # get constant value

            if error_mode == ErrorMethod.constant_dev:
                e = const_e
            elif error_mode == ErrorMethod.rss_dev_epoch:
                _, rms_e = ts_combine(ts, es, nsigma=nsigma, error_method=ErrorMethod.rms)
                sqr_error = np.vstack([rms_e, const_e]) ** 2.
                e = np.sqrt(np.sum(sqr_error, axis=0))
            elif error_mode == ErrorMethod.max_error:
                _, rms_e = ts_combine(ts, es, nsigma=nsigma, error_method=ErrorMethod.rms) # FIXME: change to RMS
                both = np.vstack([e, rms_e])
                e = np.max(both, axis=0)
        else:
            _, e = ts_combine(ts, es, nsigma=nsigma, error_method=error_mode)

        print(t.size, m.size, e.size, error_mode)
        trunc = self.get_truncation_margins()

        return WorkingMassRateDataSeries(
            None, u_gp, d_gp, b_gp, b_id, b_a, t, None, m, e,
            contributions=count, truncate=trunc
        )

    def sum(self, error_method=ErrorMethod.sum) -> WorkingMassRateDataSeries:
        if not self.series:
            return None
        elif len(self.series) == 1:
            return self.series[0]

        b_id = self.series[0].basin_id
        b_gp = self.series[0].basin_group
        b_a = self.series[0].basin_area

        u_gp = self.series[0].user_group
        d_gp = self.series[0].data_group

        for series in self:
            if series.basin_id != b_id:
                b_id = None
            if series.basin_group != b_gp:
                b_gp = None
            if series.basin_area != b_a:
                b_a = None
            if series.user_group != u_gp:
                u_gp = None
            if series.data_group != d_gp:
                d_gp = None

        ts = [series.t for series in self]
        ms = [series.dmdt for series in self]
        es = [series.errs for series in self]

        _, e = ts_combine(ts, es, error_method=error_method)
        t, m, o = sum_series(ts, ms, ret_mask=True)
        e = e[o]

        if any(s.trunc_extent is None for s in self):
            trunc = None
        else:
            avg_min_margin = np.mean([
                abs(s.trunc_extent[0] - s.t.min()) for s in self
            ])
            avg_max_margin = np.mean([
                abs(s.trunc_extent[1] - s.t.max()) for s in self
            ])
            trunc = t.min() + avg_min_margin, t.max() - avg_max_margin

        computed = any([series.computed for series in self])
        merged = any([series.merged for series in self])
        aggr = any([series.aggregated for series in self])
        count = sum([series.contributions for series in self])

        return WorkingMassRateDataSeries(
            None, u_gp, d_gp, b_gp, b_id, b_a, t, None, m, e,
            computed=computed, merged=merged, aggregated=aggr,
            contributions=count, truncate=trunc
        )

    def get_truncation_margins(self):
        """
        returns overall truncation margins for the collection
        """
        truncs = [series.trunc_extent for series in self]
        if any(map(lambda a: a is None, truncs)):
            return None
        
        t_min = min(map(lambda p: p[0], truncs))
        t_max = max(map(lambda p: p[1], truncs))
        trunc = t_min, t_max

        return trunc

    def integrate(self, offset=None, align=None) -> "model.collections.MassChangeCollection":
        out = model.collections.MassChangeCollection()
        for series in self:
            out.add_series(series.integrate(offset=offset, align=align))
        return out

    def smooth(self, window=None, clip=False, iters=1) -> "WorkingMassRateCollection":
        if window is None:
            return self
        out = WorkingMassRateCollection()
        for s in self:
            out.add_series(s.smooth(window=window, clip=clip, iters=iters))
        return out

    def filter(self, **kwargs) -> "WorkingMassRateCollection":
        return super().filter(**kwargs)

    def chunk_series(self):
        return self

    def window_cropped(self) -> "WorkingMassRateCollection":
        out = WorkingMassRateCollection()
        for s in self:
            out.add_series(s.get_truncated())
        return out

    def reduce(self, interval: float=1., centre=None, backfill=False):
        out = WorkingMassRateCollection()
        for s in self:
            out.add_series(s.reduce(interval=interval, centre=centre, backfill=backfill))
        return out

    def __add__(self, other: "WorkingMassRateCollection") -> "WorkingMassRateCollection":
        return WorkingMassRateCollection(*(self.series + other.series))

    def __iadd__(self, other: "WorkingMassRateCollection") -> "WorkingMassRateCollection":
        self.series += other.series
        return self

    def get_window(self, min_time: float, max_time: float, interp: bool=True) -> "WorkingMassRateCollection":
        assert min_time < max_time

        return WorkingMassRateCollection(
            *[s.truncate(min_time, max_time, interp=interp) for s in self if
                np.any(s.t < max_time) and np.any(s.t > min_time)]
        )

    def common_period(self):
        cp_beg = np.max([s.min_time for s in self])
        cp_end = np.min([s.max_time for s in self])

        if cp_end < cp_beg:
            return None, None
        return cp_beg, cp_end

    def min_rate(self) -> float:
        return np.min([s.min_rate for s in self])
    
    def max_rate(self) -> float:
        return np.max([s.max_rate for s in self])
    
    def min_error(self) -> float:
        return np.min([s.min_error for s in self])
    
    def max_error(self) -> float:
        return np.max([s.max_error for s in self])

    def min_rate_time(self) -> float:
        min_s = None
        for s in self:
            if min_s is None:
                min_s = s
                continue
            if s.min_rate < min_s.min_rate:
                min_s = s
        
        return s.min_rate_time()
    
    def max_rate_time(self) -> float:
        max_s = None
        for s in self:
            if max_s is None:
                max_s = s
                continue
            if s.max_rate < max_s.max_rate:
                max_s = s
        
        return s.max_rate_time()

    def stdev(self) -> float:
        return np.nanstd(
            np.hstack([s.dmdt for s in self.series])
        )

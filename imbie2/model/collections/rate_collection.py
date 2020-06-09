from .collection import Collection
from imbie2.const.error_methods import ErrorMethod
from imbie2.const.average_methods import AverageMethod
from imbie2.model.series import WorkingMassRateDataSeries, MassRateDataSeries
from imbie2.util.combine import weighted_combine as ts_combine
from imbie2.util.sum_series import sum_series
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
                error_mode: ErrorMethod=None, export_data: str=None) -> WorkingMassRateDataSeries:
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
        _, e = ts_combine(ts, es, nsigma=nsigma, error_method=error_mode)

        return WorkingMassRateDataSeries(
            None, u_gp, d_gp, b_gp, b_id, b_a, t, None, m, e,
            contributions=count
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
            out.add_series(series.integrate(offset=offset))

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

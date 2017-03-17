from .collection import Collection
from imbie2.const.error_methods import ErrorMethod
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

    def average(self, mode=None) -> WorkingMassRateDataSeries:
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

        t, m = ts_combine(ts, ms)  # TODO: averaging modes
        _, e = ts_combine(ts, es, error=True)

        return WorkingMassRateDataSeries(
            None, u_gp, d_gp, b_gp, b_id, b_a, t, None, m, e
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

        if error_method == ErrorMethod.rms:
            es = [series.errs for series in self]

            _, e = ts_combine(ts, es, error=True)
            t, m = sum_series(ts, ms)
        elif error_method == ErrorMethod.rss:
            es = [series.errs for series in self]

            _, e = sum_series(ts, es)
            t, m = sum_series(ts, ms)

            e = np.sqrt(e)
        elif error_method == ErrorMethod.sum:
            es = [series.errs for series in self]

            _, e = sum_series(ts, es)
            t, m = sum_series(ts, ms)
        else:
            raise ValueError("unknown error computation method: \"{}\"".format(error_method))

        return WorkingMassRateDataSeries(
            None, u_gp, d_gp, b_gp, b_id, b_a, t, None, m, e
        )

    def integrate(self, offset=None) -> "model.collections.MassChangeCollection":
        out = model.collections.MassChangeCollection()
        for series in self:
            out.add_series(series.integrate(offset=offset))

        return out

    def filter(self, **kwargs) -> "WorkingMassRateCollection":
        return super().filter(**kwargs)

    def chunk_series(self):
        return self

    def __add__(self, other: "WorkingMassRateCollection") -> "WorkingMassRateCollection":
        return WorkingMassRateCollection(*(self.series + other.series))

    def __iadd__(self, other: "WorkingMassRateCollection") -> "WorkingMassRateCollection":
        self.series += other.series
        return self
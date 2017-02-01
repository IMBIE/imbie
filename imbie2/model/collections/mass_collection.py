from .collection import Collection
from imbie2.model.series import MassChangeDataSeries
from imbie2.util.combine import weighted_combine as ts_combine
from imbie2.util.sum_series import sum_series
import imbie2.model as model

from typing import Iterator


class MassChangeCollection(Collection):
    def __iter__(self) -> Iterator[MassChangeDataSeries]:
        return super().__iter__()

    def combine(self) -> MassChangeDataSeries:
        b_id = self.series[0].basin_id
        b_st = self.series[0].basin_group
        b_a = self.series[0].basin_area

        ts = [series.t for series in self.series]
        ms = [series.mass for series in self.series]
        es = [series.errs for series in self.series]

        t, m = ts_combine(ts, ms)
        _, e = ts_combine(ts, es, error=True)

        return MassChangeDataSeries(
            None, None, None, b_st, b_id, b_a, t, None, m, e
        )

    def average(self, mode=None) -> MassChangeDataSeries:
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

        ts = [series.t for series in self.series]
        ms = [series.mass for series in self.series]
        es = [series.errs for series in self.series]

        t, m = ts_combine(ts, ms) # TODO: averaging modes
        _, e = ts_combine(ts, es, error=True)

        return MassChangeDataSeries(
            None, u_gp, d_gp, b_gp, b_id, b_a, t, None, m, e
        )

    def sum(self) -> MassChangeDataSeries:
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

        ts = [series.t for series in self.series]
        ms = [series.mass for series in self.series]
        es = [series.errs for series in self.series]

        t, m = sum_series(ts, ms)
        _, e = sum_series(ts, es)

        return MassChangeDataSeries(
            None, u_gp, d_gp, b_gp, b_id, b_a, t, None, m, e
        )

    def differentiate(self) -> "model.collections.WorkingMassRateCollection":
        out = model.collections.WorkingMassRateCollection()
        for series in self:
            out.add_series(series.differentiate())
        return out

    def filter(self, **kwargs) -> "MassChangeCollection":
        return super().filter(**kwargs)

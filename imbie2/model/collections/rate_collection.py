from .collection import Collection
from imbie2.model.series import WorkingMassRateDataSeries
from imbie2.util.combine import weighted_combine as ts_combine
from imbie2.util.sum_series import sum_series
import imbie2.model as model


class MassRateCollection(Collection):
    def average(self, mode=None):
        raise NotImplementedError("method not available for raw dm/dt data (chunk this collection first)")

    def sum(self):
        raise NotImplementedError("method not available for raw dm/dt data (chunk this collection first)")

    def chunk_series(self):
        out = WorkingMassRateCollection()
        for series in self:
            chunked = series.chunk_rates()
            out.add_series(chunked)
        return out

    def integrate(self):
        out = model.collections.MassChangeCollection()
        for series in self:
            out.add_series(series.integrate())

        return out

class WorkingMassRateCollection(Collection):
    def average(self, mode=None):
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

    def sum(self):
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

        _, e = sum_series(ts, es)
        t, m = sum_series(ts, ms)

        return WorkingMassRateDataSeries(
            None, u_gp, d_gp, b_gp, b_id, b_a, t, None, m, e
        )

    def integrate(self, offset=None):
        out = model.collections.MassChangeCollection()
        for series in self:
            out.add_series(series.integrate(offset=offset))

        return out
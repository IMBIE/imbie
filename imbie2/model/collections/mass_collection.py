from .collection import Collection
from imbie2.model.series import MassChangeDataSeries
from imbie2.util.functions import ts_combine


class MassChangeCollection(Collection):
    def combine(self):
        b_id = self.series[0].basin_id
        b_st = self.series[0].basin_group
        b_a = self.series[0].basin_area

        ts = [series.t for series in self.series]
        ms = [series.dM for series in self.series]
        es = [series.dM_err for series in self.series]

        t, m = ts_combine(ts, ms)
        _, e = ts_combine(ts, es, error=True)

        return MassChangeDataSeries(
            None, None, None, b_st, b_id, b_a, t, None, m, e
        )

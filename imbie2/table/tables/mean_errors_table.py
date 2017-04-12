from .base_table import Table

from imbie2.model.collections import WorkingMassRateCollection
from imbie2.const.basins import IceSheet

from typing import Callable


class MeanErrorsTable(Table):
    """
    This table shows the mean and RMS error of each dM/dt data series
    per participant and ice-sheet

    headings:
    Contributor | Method | AIS | EAIS | WAIS | APIS | GRIS
    """
    def __init__(self, data: WorkingMassRateCollection, **kwargs):
        super().__init__(**kwargs)

        users = list({s.user for s in data})
        self.add_primary_column("Contributor", "user", users)
        self.add_auto_column("Method", lambda s: s.first().user_group)

        for sheet in [IceSheet.ais, IceSheet.eais, IceSheet.wais, IceSheet.apis, IceSheet.gris]:
            self.add_auto_column(self._sheet_names[sheet], self._get_icesheet_data(sheet))

        self.generate(data)
        self.sortby = "Method"

    @staticmethod
    def _get_icesheet_data(sheet: IceSheet) -> Callable:
        def func(data: WorkingMassRateCollection) -> str:
            series = data.filter(basin_id=sheet).average()
            if series is not None:
                num = "{:.2f}\u00B1{:.2f}".format(series.mean, series.sigma)
                if series.merged:
                    num += "*"
                if series.aggregated:
                    num += "\u2020"
                return num
            else:
                return ""

        return func
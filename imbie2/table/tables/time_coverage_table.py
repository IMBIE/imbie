from .base_table import Table, Collection

from imbie2.const.basins import IceSheet

from typing import Callable


class TimeCoverageTable(Table):
    """
    This table shows the start and end time of contributions for each user.
    Unlike the MeanErrorsTable, this doesn't have a 'Method' column, as it
    is anticipated that one table will be generated per experiment group.
    """

    def __init__(self, data: Collection, **kwargs):
        super().__init__(**kwargs)

        users = list({s.user for s in data})
        self.add_primary_column("Contributor", "user", users)

        for sheet in [IceSheet.ais, IceSheet.eais, IceSheet.wais, IceSheet.apis, IceSheet.gris]:
            self.add_auto_column(self._sheet_names[sheet], self._get_icesheet_data(sheet))

        self.generate(data)

    @staticmethod
    def _get_icesheet_data(sheet: IceSheet) -> Callable:
        def func(data: Collection) -> str:
            series = data.filter(basin_id=sheet).average()
            if series is not None:
                num = "{:.2f}-\n{:.2f}".format(series.min_time, series.max_time)
                return num
            else:
                return "\n"

        return func

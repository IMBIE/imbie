from typing import Callable, Sequence, Tuple

from .base_table import Table
from imbie2.model.collections import WorkingMassRateCollection
from imbie2.const.basins import IceSheet

class RegionAveragesTable(Table):
    _region_names = {
        IceSheet.apis: "APIS",
        IceSheet.eais: "EAIS",
        IceSheet.wais: "WAIS",
        IceSheet.gris: "GrIS",
        IceSheet.ais: "AIS",
        IceSheet.all: "GrIS + AIS"
    }

    def __init__(self, data: WorkingMassRateCollection, regions: Sequence[IceSheet],
                 *times: Tuple[float, float], **kwargs):
        super().__init__(**kwargs)

        self.add_primary_column("Region", "basin_id", regions)

        for start, end in times:
            col_name = "{}-{}\n(Gt/year)".format(int(start), int(end))
            self.add_auto_column(col_name, self._get_time_data(start, end))

        self.generate(data)

    def _format_primary(self, value: IceSheet) -> str:
        return self._region_names[value]

    @staticmethod
    def _get_time_data(start: float, end: float) -> Callable:
        def func(data: WorkingMassRateCollection) -> str:
            series = data.first().truncate(start, end, interp=False)
            return "{:.2f} \u00B1 {:.2f}".format(series.mean, series.sigma)

        return func



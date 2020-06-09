from typing import Sequence, Callable

from .base_table import Table
from imbie2.const.basins import IceSheet
from imbie2.model.collections import WorkingMassRateCollection


class RegionGroupAveragesTable(Table):
    _region_names = {
        IceSheet.apis: "APIS",
        IceSheet.eais: "EAIS",
        IceSheet.wais: "WAIS",
        IceSheet.gris: "GrIS",
        IceSheet.ais: "AIS",
        IceSheet.all: "GrIS + AIS"
    }

    def __init__(self, group_data: WorkingMassRateCollection, cross_data: WorkingMassRateCollection,
                 regions: Sequence[IceSheet], min_date: float, max_date: float, groups: Sequence[str],**kwargs):
        super().__init__(**kwargs)

        self.add_primary_column("Region", "basin_id", regions)
        self.min_date = min_date
        self.max_date = max_date

        for group in groups:
            name = "{} mass balance (Gt/yr)".format('MB' if group == 'IOM' else group)
            self.add_auto_column(name, self._get_group_values(group))
        self.add_auto_column("Average mass balance (Gt/yr)", self._get_cross_values)

        self.generate(group_data+cross_data)

    def _format_primary(self, value: IceSheet) -> str:
        return self._region_names[value]

    def _get_group_values(self, group: str) -> Callable:
        def func(data: WorkingMassRateCollection) -> str:
            series = data.filter(user_group=group).first().truncate(self.min_date, self.max_date)
            return "{:.0f} \u00B1 {:.0f}".format(series.mean, series.sigma)

        return func

    def _get_cross_values(self, data: WorkingMassRateCollection) -> str:
        series = data.filter(user_group=None).first().truncate(self.min_date, self.max_date)
        return "{:.0f} \u00B1 {:.0f}".format(series.mean, series.sigma)

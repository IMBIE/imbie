from .base_table import Table, Collection

from imbie2.const.basins import IceSheet, BasinGroup, ZwallyBasin, RignotBasin, Basin

from typing import Callable


class BasinsTable(Table):
    """

    """
    def __init__(self, data: Collection, basin_group: BasinGroup, **kwargs):
        super().__init__(**kwargs)

        data = data.filter(basin_group=basin_group)
        users = list({s.user for s in data})
        self.add_primary_column("Contributor", "user", users)

        basin_set = {
            BasinGroup.zwally: ZwallyBasin,
            BasinGroup.rignot: RignotBasin,
            BasinGroup.sheets: IceSheet
        }

        for basin in basin_set[basin_group]:
            self.add_auto_column(basin.value, self._get_basin_data(basin))
        for sheet in [IceSheet.ais, IceSheet.eais, IceSheet.wais, IceSheet.apis, IceSheet.gris]:
            self.add_auto_column(sheet.value, self._get_basin_data(sheet))

        self.generate(data)

    @staticmethod
    def _get_basin_data(basin: Basin) -> Callable:
        def func(data: Collection) -> str:
            series = data.filter(basin_id=basin).average()
            return "\u2714" if series is not None else ""

        return func

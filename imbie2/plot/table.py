from prettytable import PrettyTable
from collections import OrderedDict
from typing import Sequence, Any, Union

from imbie2.const.basins import IceSheet
from imbie2.model.collections import MassChangeCollection, WorkingMassRateCollection

Collection = Union[MassChangeCollection, WorkingMassRateCollection]

class Table(PrettyTable):
    _sheet_names = {
        IceSheet.ais: "Antarctica",
        IceSheet.wais: "West Antarctica",
        IceSheet.eais: "East Antarctica",
        IceSheet.apis: "Antarctic Peninsula",
        IceSheet.gris: "Greenland"
    }

    def __init__(self, *field_names, **kwargs):
        super().__init__(field_names=field_names, **kwargs)
        self._auto_cols = OrderedDict()

        self._primary_vals = []
        self._primary_attr = None

    def add_primary_column(self, name: str, attr: str, values: Sequence[Any]):
        self._primary_attr = attr
        self._primary_vals = values
        self.add_column(name, values)

    def _retreive_primary(self, data: Collection):
        for val in self._primary_vals:
            filter = {self._primary_attr: val}
            yield data.filter(**filter)

    def add_auto_column(self, name: str, function):
        self._auto_cols[name] = function

    def generate(self, data: Collection):
        for name, func in self._auto_cols.items():
            col = [func(item) for item in self._retreive_primary(data)]
            self.add_column(name, col)


class MeanErrorsTable(Table):
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
    def _get_icesheet_data(sheet: IceSheet):
        def func(data: WorkingMassRateCollection):
            series = data.filter(basin_id=sheet).average()
            if series is not None:
                num = "{:.2f}+/-{:.2f}".format(series.mean, series.sigma)
                if series.merged:
                    num += "*"
                return num
            else:
                return ""

        return func


class TimeCoverageTable(Table):
    def __init__(self, data: WorkingMassRateCollection, **kwargs):
        super().__init__(**kwargs)

        users = list({s.user for s in data})
        self.add_primary_column("Contributor", "user", users)

        for sheet in [IceSheet.ais, IceSheet.eais, IceSheet.wais, IceSheet.apis, IceSheet.gris]:
            self.add_auto_column(self._sheet_names[sheet], self._get_icesheet_data(sheet))

        self.generate(data)

    @staticmethod
    def _get_icesheet_data(sheet: IceSheet):
        def func(data: WorkingMassRateCollection):
            series = data.filter(basin_id=sheet).average()
            if series is not None:
                num = "{:.2f}-\n{:.2f}".format(series.min_time, series.max_time)
                return num
            else:
                return "\n"

        return func

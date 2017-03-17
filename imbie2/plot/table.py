from prettytable import PrettyTable, MSWORD_FRIENDLY
from collections import OrderedDict
from typing import Sequence, Any, Union, Iterator, Callable

from imbie2.const.basins import IceSheet, BasinGroup, ZwallyBasin, RignotBasin, Basin
from imbie2.model.collections import MassChangeCollection, WorkingMassRateCollection

Collection = Union[MassChangeCollection, WorkingMassRateCollection]


class Table(PrettyTable):
    """
    Base class for IMBIE tables, extends prettytable.PrettyTable with
    methods to automatically create table contents from a dataset.

    NB: I considered making this an abstract base class, but decided against
     it since it could be useful to create some weird one-off tables if that's
     a thing that we need to do.
     Nonetheless, it would be preferable to create a new sub-class for any new
     tables similar to the others currently in this file.
    """

    _sheet_names = {
        IceSheet.ais: "Antarctica",
        IceSheet.wais: "West Antarctica",
        IceSheet.eais: "East Antarctica",
        IceSheet.apis: "Antarctic Peninsula",
        IceSheet.gris: "Greenland"
    }

    def __init__(self, *field_names, **kwargs):
        word_mode = kwargs.pop('msword', False)

        super().__init__(field_names=field_names, **kwargs)
        if word_mode:
            self.set_style(MSWORD_FRIENDLY)

        self._auto_cols = OrderedDict()

        self._primary_vals = []
        self._primary_attr = None

    def add_primary_column(self, name: str, attr: str, values: Sequence[Any]) -> None:
        """
        creates the primary column. This requires an attribute and set of values
        with which the dataset can be filtered - the data in the other columns will
        be generated from the results of these filters.

        :param name: The column heading
        :param attr: The DataSeries property against which to filter (eg: user)
        :param values: The set of values for the column
        """
        self._primary_attr = attr
        self._primary_vals = values
        self.add_column(name, values)

    def _retreive_primary(self, data: Collection) -> Iterator[Collection]:
        """
        filters the input dataset for each value in the primary column

        :param data: The input collection of data
        :return: an iterator of filter results
        """
        for val in self._primary_vals:
            filter = {self._primary_attr: val}
            yield data.filter(**filter)

    def add_auto_column(self, name: str, function: Callable) -> None:
        """
        add a new automatically-generated column to the table

        :param name: the heading of the column
        :param function: a function to produce column values from input data
        """
        self._auto_cols[name] = function

    def generate(self, data: Collection) -> None:
        """
        creates all the automatic columns for the table.

        :param data: the input data collection
        """
        for name, func in self._auto_cols.items():
            col = [func(item) for item in self._retreive_primary(data)]
            self.add_column(name, col)


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

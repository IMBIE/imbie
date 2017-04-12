from prettytable import PrettyTable, MSWORD_FRIENDLY
from collections import OrderedDict
from typing import Sequence, Any, Union, Iterator, Callable

from imbie2.const.basins import IceSheet, Basin
from imbie2.const.table_formats import TableFormat
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

    def __init__(self, *field_names, style: TableFormat=TableFormat.fancy, **kwargs):
        super().__init__(field_names=field_names, **kwargs)

        if style == TableFormat.msword:
            self.set_style(MSWORD_FRIENDLY)
        self._format = style

        self._auto_cols = OrderedDict()

        self._primary_vals = []
        self._primary_attr = None

    def write(self, filename: str, **kwargs) -> None:
        """
        write the table to a file

        :param filename: the path to the file
        """
        with open(filename, 'w') as f:
            f.write(self.get_string(**kwargs))

    def default_extension(self) -> str:
        """
        get the default file extension for using to
        save this table
        """
        if self._format == TableFormat.html:
            return "html"
        if self._format == TableFormat.csv:
            return "csv"
        return "txt"

    def get_string(self, **kwargs) -> str:
        if self._format == TableFormat.html:
            return self.get_html_string()

        elif self._format == TableFormat.csv:
            # create header line
            string = ",".join([field for field in self._get_field_names()]) + "\n"

            # iterate over rows of data
            for row in self:
                # remove headers and borders
                row.border = False
                row.header = False

                # create comma-separated line of data
                line = ",".join(
                    [row.get_string(fields=[field]).strip().replace('\n', '') for
                       field in self._get_field_names()]
                )
                string += line + "\n"
            return string

        else:
            return super().get_string(**kwargs)

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

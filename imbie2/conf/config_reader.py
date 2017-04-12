from typing import Any
from collections import OrderedDict
from ast import literal_eval
from warnings import warn

from .config_errors import *
from .config_param import ConfigParam

from imbie2.const.groups import Group
from imbie2.const.average_methods import AverageMethod
from imbie2.const.table_formats import TableFormat


class ConfigFile:

    @classmethod
    def _get_parameters(cls) -> Sequence[str]:
         return (
             item.name for name, item in vars(cls).items() if\
                isinstance(item, ConfigParam)
         )

    def _get_value(self, param_name: str) -> Any:
        if param_name not in self._data:
            raise MissingParameterError(param_name)
        return self._data[param_name]

    def __init__(self, fname: str):
        self._data = OrderedDict()
        self._file = None
        self.filepath = fname

    def __enter__(self) -> "ConfigFile":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        self._file = open(self.filepath)
        self.read(self._file)

    def read(self, fileobj) -> None:
        for line in fileobj:

            line_data = line.strip().split()

            if not line_data:
                continue

            name = line_data.pop(0)

            value = list(map(literal_eval, line_data))

            if name in self._get_parameters():
                self._data[name] = value
            else:
                warn(UnknownParameterWarning(name))

    def close(self) -> None:
        self._file.close()


class ImbieConfig(ConfigFile):
    input_path = ConfigParam("input_path", str)
    output_path = ConfigParam("output_path", str)

    plot_format = ConfigParam("plot_format", str, options=["png", "jpg", "svg", "pdf"], optional=True)
    table_format = ConfigParam("table_format", TableFormat, default=TableFormat.fancy)

    start_date = ConfigParam("start_date", float, optional=True)
    stop_date = ConfigParam("stop_date", float, optional=True)

    methods_skip = ConfigParam("methods_skip", Group, multiple=True)
    users_skip = ConfigParam("users_skip", str, multiple=True)

    combine_method = ConfigParam("combine_method", AverageMethod, default=AverageMethod.equal_groups)
    align_date = ConfigParam("align_date", float, optional=True)

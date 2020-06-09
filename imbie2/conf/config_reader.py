from typing import Any
from collections import OrderedDict
from ast import literal_eval
from warnings import warn

from .config_errors import *
from .config_param import ConfigParam

from imbie2.const.groups import Group
from imbie2.const.average_methods import AverageMethod
from imbie2.const.error_methods import ErrorMethod
from imbie2.const.table_formats import TableFormat
from imbie2.const.lsq_methods import LSQMethod

import matplotlib.pyplot as plt
format_opts = list(plt.gcf().canvas.get_supported_filetypes().keys())


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

    plot_format = ConfigParam("plot_format", str, options=format_opts, optional=True)
    table_format = ConfigParam("table_format", TableFormat, default=TableFormat.fancy)

    start_date = ConfigParam("start_date", float, optional=True)
    stop_date = ConfigParam("stop_date", float, optional=True)

    methods_skip = ConfigParam("methods_skip", Group, multiple=True)
    users_skip = ConfigParam("users_skip", str, multiple=True)
    users_mark = ConfigParam("users_mark", str, multiple=True)

    combine_method = ConfigParam("combine_method", AverageMethod, default=AverageMethod.equal_groups)
    group_avg_errors_method = ConfigParam("group_avg_error_method", ErrorMethod, optional=True)
    sheet_avg_errors_method = ConfigParam("sheet_avg_error_method", ErrorMethod, optional=True)
    sum_errors_method = ConfigParam("sum_errors_method", ErrorMethod, default=ErrorMethod.sum)

    align_date = ConfigParam("align_date", float, optional=True)
    average_nsigma = ConfigParam("average_nsigma", float, optional=True)
    plot_smooth_window = ConfigParam("plot_smooth_window", float, optional=True)
    plot_smooth_iters = ConfigParam("plot_smooth_iters", int, optional=True)

    export_data = ConfigParam("export_data", bool, default=False)
    include_la = ConfigParam("enable_la_group", bool, default=False)

    bar_plot_min_time = ConfigParam("bar_plot_min_time", float, optional=True)
    bar_plot_max_time = ConfigParam("bar_plot_max_time", float, optional=True)

    # params from dm-to-dmdt conversion
    use_dm = ConfigParam("use_dm", bool, default=False)
    dmdt_window = ConfigParam("dmdt_window", float, default=1.)
    dmdt_method = ConfigParam("dmdt_method", LSQMethod, default=LSQMethod.normal)
    truncate_dmdt = ConfigParam("truncate_dmdt", bool, default=True)
    truncate_avg = ConfigParam("truncate_avg", bool, default=False)
    apply_dmdt_smoothing = ConfigParam('apply_dmdt_smoothing', bool, default=True)

    reduce_window = ConfigParam("reduce_window", float, optional=True)
    data_smoothing_window = ConfigParam("data_smoothing_window", float, optional=True)
    data_smoothing_iters = ConfigParam("data_smoothing_iters", int, optional=True)
    export_smoothing_window = ConfigParam("export_smoothing_window", float, optional=True)
    export_smoothing_iters = ConfigParam("export_smoothing_iters", int, optional=True)
    imbie1_compare = ConfigParam("imbie1_compare", bool, default=True)

    output_timestep = ConfigParam("output_timestep", float, optional=True)
    output_offset = ConfigParam("output_offset", float, optional=True)

    def read(self, fileobj) -> None:
        super().read(fileobj)

        if self.group_avg_errors_method is None:
            if self.combine_method == AverageMethod.imbie1_compat:
                self.group_avg_errors_method = ErrorMethod.imbie1
            else:
                self.group_avg_errors_method = ErrorMethod.rms

        if self.sheet_avg_errors_method is None:
            if self.combine_method == AverageMethod.imbie1_compat:
                self.sheet_avg_errors_method = ErrorMethod.imbie1
            else:
                self.sheet_avg_errors_method = ErrorMethod.rms

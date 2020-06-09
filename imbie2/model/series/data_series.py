from abc import ABCMeta, abstractmethod
from imbie2.const.basins import Basin, BasinGroup
from typing import Optional


class DataSeries(metaclass=ABCMeta):
    @property
    def min_time(self) -> float:
        return self._get_min_time()

    @property
    def max_time(self) -> float:
        return self._get_max_time()

    def __init__(self, user: Optional[str], user_group: Optional[str], data_group: Optional[str],
                 basin_group: BasinGroup, basin_id: Basin, basin_area: float, computed: bool=False, merged: bool=False,
                 aggregated: bool=False, contributions: int=1):
        # name of user
        self.user = user
        # experiment group
        self.user_group = user_group
        # experiment group (from data files)
        self.data_group = data_group
        # zwally/rignot/generic
        self.basin_group = basin_group
        # basin/ice-sheet id
        self.basin_id = basin_id
        # full basin area
        self.basin_area = basin_area

        # True if series has been converted from dM/dt or dM
        self.computed = computed
        # True if series has been merged from Rignot & Zwally
        self.merged = merged
        # True if series has been computed from indiv. basins
        self.aggregated = aggregated
        # number of contributing data series
        self.contributions = contributions

    def limit_times(self, min_t: float=None, max_t: float=None, interp: bool=True) -> None:
        if min_t is not None:
            self._set_min_time(min_t, interp=interp)
        if max_t is not None:
            self._set_max_time(max_t, interp=interp)

    @abstractmethod
    def _set_min_time(self, min_t: float, interp: bool=True) -> None:
        return

    @abstractmethod
    def _set_max_time(self, max_t: float, interp: bool=True) -> None:
        return

    @abstractmethod
    def _get_min_time(self) -> float:
        return

    @abstractmethod
    def _get_max_time(self) -> float:
        return

    def temporal_resolution(self) -> float:
        return (self.max_time - self.min_time) / len(self)

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
                 aggregated: bool=False):
        self.user = user
        self.user_group = user_group
        self.data_group = data_group
        self.basin_group = basin_group
        self.basin_id = basin_id
        self.basin_area = basin_area

        self.computed = computed
        self.merged = merged
        self.aggregated = aggregated

    def limit_times(self, min_t: float=None, max_t: float=None) -> None:
        if min_t is not None:
            self._set_min_time(min_t)
        if max_t is not None:
            self._set_max_time(max_t)

    @abstractmethod
    def _set_min_time(self, min_t: float) -> None:
        return

    @abstractmethod
    def _set_max_time(self, max_t: float) -> None:
        return

    @abstractmethod
    def _get_min_time(self) -> float:
        return

    @abstractmethod
    def _get_max_time(self) -> float:
        return

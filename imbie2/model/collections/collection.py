from abc import ABCMeta, abstractmethod
from typing import Sequence, Iterable, Union

from imbie2.model.series import *
from imbie2.const.error_methods import ErrorMethod
Series = Union[MassChangeDataSeries, MassRateDataSeries, WorkingMassRateDataSeries]

class Collection(metaclass=ABCMeta):

    def __init__(self, *series: Sequence[Series]):
        self.series = list(series)

    def min_time(self) -> float:
        min_t = None
        for series in self:
            t = series.min_time
            if min_t is None or t < min_t:
                min_t = t
        return min_t

    def max_time(self) -> float:
        max_t = None
        for series in self:
            t = series.max_time
            if max_t is None or t > max_t:
                max_t = t
        return max_t

    def concurrent_start(self) -> float:
        """
        returns earliest time that all series have data
        """
        beg_t = None
        for series in self:
            t = series.min_time
            if beg_t is None or t > beg_t:
                beg_t = t
        return beg_t

    def concurrent_stop(self) -> float:
        """
        returns latest time that all series have data
        """
        end_t = None
        for series in self:
            t = series.max_time
            if end_t is None or t < end_t:
                end_t = t
        return end_t

    def add_series(self, series: Series) -> None:
        self.series.append(series)

    def merge(self) -> None:
        rem = []
        new = []
        for a in self:
            if a in rem:
                continue
            for b in self:
                if b in rem:
                    continue
                if a.user == b.user and\
                   a.user_group == b.user_group and\
                   a.basin_id == b.basin_id and\
                   a.basin_group != b.basin_group:
                    s = b.merge(b, a)
                    if s is not None:
                        rem.append(a)
                        rem.append(b)
                        new.append(s)
        for s in rem:
            self.series.remove(s)
        for s in new:
            self.series.append(s)

    def filter(self, **kwargs) -> "Collection":
        out = self.__class__()
        if "_max" in kwargs:
            _max = kwargs.pop("_max")
        else: _max = None

        for series in self:
            valid = True
            for key, expected in kwargs.items():
                val = getattr(series, key)

                if isinstance(expected, Sequence):
                    if val is None or val not in expected:
                        valid = False
                        break
                elif val != expected:
                    valid = False
                    break

            if valid:
                out.add_series(series)
                if _max is not None and len(out) >= _max:
                    break

        return out

    def first(self) -> Series:
        if not self.series:
            return None
        return self.series[0]

    @abstractmethod
    def average(self, mode=None) -> Series:
        """
        average data of series in collection
        """
        return None

    @abstractmethod
    def sum(self, error_method: ErrorMethod=ErrorMethod.sum) -> Series:
        """
        sum data of series in collection
        """
        return None

    def __iter__(self) -> Iterable[Series]:
        return iter(self.series)

    def __getitem__(self, index) -> Series:
        return self.series[index]

    def __len__(self) -> int:
        return len(self.series)

    def __bool__(self) -> bool:
        return bool(self.series)

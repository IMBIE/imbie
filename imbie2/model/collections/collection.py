from abc import ABCMeta, abstractmethod
from typing import Sequence


class Collection(metaclass=ABCMeta):

    def __init__(self, *series):
        self.series = list(series)

    def min_time(self):
        min_t = None
        for series in self:
            t = series.min_time
            if min_t is None or t < min_t:
                min_t = t
        return min_t

    def max_time(self):
        max_t = None
        for series in self:
            t = series.max_time
            if max_t is None or t > max_t:
                max_t = t
        return max_t

    def concurrent_start(self):
        """
        returns earliest time that all series have data
        """
        beg_t = None
        for series in self:
            t = series.min_time
            if beg_t is None or t > beg_t:
                beg_t = t
        return beg_t

    def concurrent_stop(self):
        """
        returns latest time that all series have data
        """
        end_t = None
        for series in self:
            t = series.max_time
            if end_t is None or t < end_t:
                end_t = t
        return end_t

    def add_series(self, series):
        self.series.append(series)

    def merge(self):
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

    def filter(self, **kwargs):
        out = self.__class__()
        for series in self:
            valid = True
            for key, expected in kwargs.items():
                val = getattr(series, key)

                if isinstance(expected, Sequence):
                    if val not in expected:
                        valid = False
                        break
                elif val != expected:
                    valid = False
                    break

            if valid:
               out.add_series(series)

        return out

    @abstractmethod
    def average(self, mode=None):
        """
        average data of series in collection
        """
        return None

    @abstractmethod
    def sum(self):
        """
        sum data of series in collection
        """
        return None

    def __iter__(self):
        return iter(self.series)

    def __getitem__(self, index):
        return self.series[index]

    def __len__(self):
        return len(self.series)

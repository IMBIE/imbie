from abc import ABCMeta, abstractmethod


class Collection(metaclass=ABCMeta):

    def __init__(self, *series):
        self.series = list(series)

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

    def __iter__(self):
        return iter(self.series)

    def __getitem__(self, index):
        return self.series[index]

    def __len__(self):
        return len(self.series)

    @abstractmethod
    def combine(self):
        """
        combine the data of the member time series
        """

from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class CollectionsManager(metaclass=ABCMeta):

    def __init__(self, min_time=None, max_time=None):
        self.collections = OrderedDict()
        self.min_t = min_time
        self.max_t = max_time

    def add_series(self, series):
        series.limit_times(self.min_t, self.max_t)

        if series.basin_id in self.collections:
            self.collections[series.basin_id].add_series(series)
        else:
            collection = self.new_collection(series)
            self.collections[series.basin_id] = collection

    def merge(self):
        for c in self:
            c.merge()

    @abstractmethod
    def new_collection(self, series):
        return None

    def __iter__(self):
        return iter(self.collections.values())

    def __len__(self):
        return len(self.collections)

    def __getitem__(self, index):
        return self.collections.get(index, [])

    def __contains__(self, index):
        return index in self.collections

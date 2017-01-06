from abc import abstractmethod
from collections import OrderedDict
from typing import Iterable

from imbie2.model.collections.collection import Collection


class CollectionsManager:

    def __init__(self, min_time=None, max_time=None):
        self.collections = OrderedDict()
        self.min_t = min_time
        self.max_t = max_time

    def add_series(self, series) -> None:
        series.limit_times(self.min_t, self.max_t)

        if series.basin_id in self.collections:
            self.collections[series.basin_id].add_series(series)
        else:
            collection = self.new_collection(series)
            self.collections[series.basin_id] = collection

    def merge(self) -> None:
        for c in self:
            c.merge()

    @abstractmethod
    def as_collection(self):
        return None

    @abstractmethod
    def new_collection(self, series):
        return None

    def __iter__(self) -> Iterable[Collection]:
        return iter(self.collections.values())

    def __len__(self) -> int:
        return len(self.collections)

    def __getitem__(self, index) -> Iterable[Collection]:
        return self.collections.get(index, [])

    def __contains__(self, index) -> bool:
        return index in self.collections

from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class Basins(metaclass=ABCMeta):
    def __init__(self):
        self._basins = OrderedDict()

    def __iter__(self):
        return self._basins.keys()

    def __len__(self):
        return len(self._basins)

    def __contains__(self, basin):
        return basin in self._basins

    def __getitem__(self, basin):
        return self._basins[basin]

class Basin:
    def __init__(self, basin_id, lats, lons):
        self.id = basin_id
        self.lats = lats
        self.lons = lons

from imbie2.model.collections import MassChangeCollection
from .manager import CollectionsManager


class MassChangeCollectionsManager(CollectionsManager):
    def new_collection(self, series):
        return MassChangeCollection(series)

    def as_collection(self):
        out = MassChangeCollection()
        for col in self:
            for series in col:
                out.add_series(series)
        return out

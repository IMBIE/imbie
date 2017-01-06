from imbie2.model.collections import MassRateCollection
from .manager import CollectionsManager


class MassRateCollectionsManager(CollectionsManager):
    def new_collection(self, series):
        return MassRateCollection(series)

    def as_collection(self):
        out = MassRateCollection()
        for col in self:
            for series in col:
                out.add_series(series)
        return out
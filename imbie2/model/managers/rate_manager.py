from imbie2.model.collections import MassRateCollection
from .manager import CollectionsManager


class MassRateCollectionsManager(CollectionsManager):
    def new_collection(self, series):
        return MassRateCollection(series)

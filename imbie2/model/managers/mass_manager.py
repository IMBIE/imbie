from imbie2.model.collections import MassChangeCollection
from .manager import CollectionsManager


class MassChangeCollectionsManager(CollectionsManager):
    def new_collection(self, series):
        return MassChangeCollection(series)

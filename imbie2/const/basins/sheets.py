import enum
from .groups import BasinGroup


class IceSheet(enum.Enum):
    # antarctica basins:
    apis = 'apis'
    wais = 'wais'
    eais = 'eais'
    # greenland basins:
    gris = 'gris'
    # all antarctica:
    ais = 'ais'

    @classmethod
    def is_valid(cls, name):
        name = name.split(':')[-1].strip()

        if name in cls.__members__:
            return True
        if name == 'gis':
            return True
        return False

    @classmethod
    def get_basin(cls, name):
        name = name.split(':')[-1].strip()

        if name == 'gis':
            name = 'gris'

        return cls(name)

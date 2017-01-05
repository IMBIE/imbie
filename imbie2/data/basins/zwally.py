import pandas as pd
from .basins import *
from imbie2.const.basins import ZwallyBasins


class ZwallyBasins(Basins):
    def __init__(self, filename, header=7):
        super().__init__()

        data = pd.read_csv(
            fname, header=None, skiprows=header,
            names=['lat', 'lon', 'ids'],
            delim_whitespace=True
        )
        ids = data.ids.unique()

        for _id in ids:
            ok = np.flatnonzero(data.ids == _id)
            lats = data.lat[ok]
            lons = data.lon[ok]

            basin = ZwallyBasin(str(_id))
            self._basins[basin] = Basin(
                basin, lats, lons
            )

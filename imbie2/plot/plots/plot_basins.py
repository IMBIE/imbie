#!/usr/bin/python3
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib as mpl
from matplotlib import cm
import pandas as pd
import sys
from cycler import cycler
from itertools import cycle
from random import Random


class BasinArtist:
    def __init__(self, ax, basemap):
        self.ax = ax
        self.bm = basemap

    def draw_poly(self, lats, lons, **style):
        lats = np.asarray(lats)
        lons = np.asarray(lons)
        xs, ys = self.bm(lons, lats)
        points = list(zip(xs, ys))

        poly = mp.Polygon(points, **style)
        self.ax.add_patch(poly)

        return np.mean(xs), np.mean(ys)

def plot_basin_map(ax, basins, data, cmap=None):
    # fname = sys.argv[1]
    # skip = int(sys.argv[2])
    # rnd = Random()
    #
    # names = ['lat', 'lon', 'ids']
    # data = pd.read_csv(
    #     fname, header=None, skiprows=skip,
    #     names=names, delim_whitespace=True
    # )
    # ids = data.ids.unique()
    #
    # fig, ax = plt.subplots()
    m = Basemap(projection='spstere',boundinglat=-65,lon_0=180)
    m.drawmapboundary(fill_color='white')
    plotter = BasinArtist(ax, m)

    style = {
        'ec': 'black',
        'ls': '-',
        'lw': .5
    }
    if cmap is None:
        cmap = cm.viridis

    max_val = None
    min_val = None
    for basin_id in basins:
        val = data[basin_id].mean
        if val > max_val or max_val is None:
            max_val = val
        if val < min_val or min_val is None:
            min_val = val

    for basin_id in basins:
        lats = basins[basin_id].lats
        lons = basins[basin_id].lons

        val = data[basin_id].mean
        norm = (val - min_val) / (max_val - min_val)

        style['color'] = cmap(norm)
        x, y = plotter.draw_poly(lats, lons, **style)
        ax.plot(x, y, 'kx')
        ax.annotate(
            basin_id.value, (x - 2.5e4, y - 2.5e4),
        )
    m.drawparallels(np.linspace(-80, -50, 4))
    m.drawmeridians(np.arange(0, 360, 20))
    m.drawcoastlines()

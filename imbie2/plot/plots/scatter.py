from imbie2.const.basins import *
from imbie2.plot.style import colours
from matplotlib import markers
import matplotlib.lines as mlines

def basin_scatter(ax, data, marker='o'):
    p = None
    groups = set()

    for series in data:
        if series.computed: continue

        c = colours.primary[series.user_group]
        groups.add(series.user_group)
        p = ax.scatter(series.t, series.mass,
                       c=c, marker=marker)
    return groups

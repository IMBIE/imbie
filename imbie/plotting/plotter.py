import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from functools import wraps
from itertools import count
import os

from ..functions import move_av
from .style import apply_style


def plot_err_poly(t, dmdt, dmdt_sd, col1='g', col2='b', average=False,
                  nsigma=10, line=False, offset=0, spacing=1, thick=None, ax=None, zrank=None):
    """
    Draws a dm/dt vs time plot with error margins.

    INPUTS:
        t:    the time series of the data
        dmdt: the dm/dt data series
        dmdt_sd: the error margins of the dm/dt data
        col1: (optional) The background color to render
                the error margins with
        col2: (optional) The foreground color to render
                the dm/dt plot with
        average: (optional) If True, perform a moving average on the input data
        nsigma: (optional) The tolerance for large changes in value
        line: (optional) If True, draw the error margin with vertical lines instead of solid fill
        offset: (optional) The first index at which to start drawing lines if 'line' is True
        spacing: (optional) The step between indices at which the draw lines if 'line' is True
        thick: (optional) The thickness of the dm/dt line
        ax: (optional) an Axes object on which to draw the plots
        zrank: (optional) The z-order position at which to draw the plots
                (higher -> foreground, lower-> background)
    """
    # if no argument for thickness, get the default thickness for the current style
    if thick is None:
        thick = mpl.rcParams["lines.linewidth"]
    # find locations where dm/dt has finite values
    ok = np.isfinite(dmdt)
    # remove locations where the value differs greatly from the mean
    ok[ok] = np.abs(dmdt[ok] - np.nanmean(dmdt)) < max(nsigma, 1) * max(np.nanstd(dmdt), .1)

    # get time, dm/dt and error vals at valid locations
    ti = t[ok]
    dmdti = dmdt[ok]
    dmdt_sdi = dmdt_sd[ok]

    # if no Axes object provided, use plt.plot etc.
    if ax is None:
        ax = plt

    # perform moving average
    if average:
        dmdti = move_av(13. / 12, ti, dmdti)
    if line:
        # line-mode plotting
        #   plot error margin above and below the dm/dt line
        for i in -1, 1:
            ax.plot(ti, dmdti + i * dmdt_sdi, color=col1, zorder=zrank)
        #   plot the vertical lines
        for i in range(offset, len(ti), spacing):
            ax.plot([ti[i], ti[i]], [dmdti[i] + dmdt_sdi[i], dmdti[i] - dmdt_sdi[i]], color=col1, zorder=zrank)
    else:
        # fill-mode plotting
        #   draw solid fill between the dmdt line +/- error margin
        ax.fill_between(ti, (dmdti - dmdt_sdi), (dmdti + dmdt_sdi), facecolor=col1, alpha=.5, zorder=zrank)
    # draw the dm/dt line
    ax.plot(ti, dmdti, color=col2, linewidth=thick, zorder=zrank+1)


class Plotter:
    """
    Class for creating IMBIE plots.
    """
    # create colours
    colors = [u'#000000', u'#ffffff', u'#c8c8c8',
              u'#ff0000', u'#0000ff', u'#00ff00',
              u'#00ffff', u'#ffc8c8', u'#c8c8ff',
              u'#c8ffc8', u'#c8ffff', u'#ffff00',
              u'#ff00ff', u'#ffc8ff', u'#ff8000',
              u'#ffdc80']

    # set colour scheme for satellites
    sat_cols = {
        "ICESat": (colors[9], colors[5]),
        "IOM": (colors[7], colors[3]),
        "RA": (colors[10], colors[6]),
        "GRACE": (colors[8], colors[4]),
        "All": (colors[1], colors[2])
    }
    # set colour scheme for ice sheets
    sheet_cols = {
        "EAIS": (colors[9], colors[5]),
        "WAIS": (colors[8], colors[4]),
        # "APIS": (colors[7], colors[3]), old value of APIS - changed to match original graphs
        "APIS": (colors[15], colors[14]),
        "GrIS": (colors[10], colors[6]),
        "AIS": (colors[13], colors[12]),
        "AIS+GrIS": (colors[2], colors[0])
    }
    long_names = {
        "IOM": "Input output method",
        "RA": "Radar altimetry",
        "GRACE": "Gravimetry",
        "ICESat": "Laser altimetry",

        "EAIS": "East Antarctica",
        "WAIS": "West Antarctica",
        "APIS": "Antarctic Peninsula",
        "GrIS": "Greenland",
        "AIS": "Antarctica",
        "AIS+GrIS": "Antarctica & Greenland"
    }

    def __init__(self, file_type=None, path=None, **fig_config):
        """
        Initializes the Plotter instance.
        INPUTS:
            file_type: (optional) the format to save plots as. Any format
                        supported by matplotlib may be used:
                            "svg", "png", "pdf", "jpg", ...
                        if the argument is not supplied, plots are instead
                        shown in the user interface via the plt.show() function.
            path: (optional) the directory into which to save the plots. If this
                  argument is not supplied, the program's current working directory
                  is used.
        """
        self.mode = file_type
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path
        apply_style(**fig_config)

    def output_plot(self, fname):
        """
        saves or displays the current plot
        """
        plt.minorticks_on()
        if self.mode is None:
            plt.show()
        else:
            if not os.path.exists(self.path):
                os.mkdir(self.path)

            fname = "{}.{}".format(fname, self.mode)
            path = os.path.join(self.path, fname)

            plt.savefig(path, border_inches='tight')
        plt.close()

    def plot_dmdt_ice_sheet_method(self, dmdt, dmdt_sd, sheet_order=None, source_order=None):
        if sheet_order is None:
            sheet_order = dmdt.keys()
        if source_order is None:
            source_order = dmdt[sheet_order[0]].keys()
            source_order.remove('All')

        fig, ax = plt.subplots()

        n_sheets = len(sheet_order)
        n_sources = len(source_order)

        tick_locs = np.arange(n_sources/2, n_sources*(n_sheets+1), n_sources)
        plt.xticks(tick_locs, sheet_order)

        plt.xlabel("Ice Sheet")
        plt.ylabel("Mass balance (Gt/yr)")

        plt.xlim(-1, n_sources*n_sheets + 1)
        plt.tick_params(axis='x', which='minor', bottom='off', top='off')

        verts = []
        patches = []

        xpad = np.array([.25, .25, .75, .75])
        for i, sheet_id in enumerate(sheet_order):
            x = np.array([0, 0, n_sources-1, n_sources-1]) + i * n_sources + xpad
            y = np.array([-1, 1, 1, -1]) * dmdt_sd[sheet_id]['All'] + dmdt[sheet_id]['All']
            verts.append(zip(x, y))
            for j, source_id in enumerate(source_order):
                x = i * n_sources + j + .5
                y = dmdt[sheet_id][source_id]
                yerr = dmdt_sd[sheet_id][source_id]

                _, color = self.sat_cols[source_id]
                plt.errorbar(x, y, yerr=yerr, ecolor=color)

        for source_id in source_order:
            _, color = self.sat_cols[source_id]
            patches.append(
                mpatches.Patch(color=color, label=source_id)
            )
        # create legend
        l = plt.legend(handles=patches)
        # colorize text of items in legend
        for text, patch in zip(l.get_texts(), l.get_patches()):
            # get the color of the patch
            col = patch.get_facecolor()
            # make the patch invisible
            # patch.set_fill(False)
            # set colour of text to colour of patch
            text.set_color(col)

        # create a collection from the polygons
        _, color =self.sat_cols['All']
        col = PolyCollection(verts, facecolors=[color for _ in verts])
        # add the polygons to the plot
        ax.add_collection(col)

        self.output_plot("dmdt_ice_sheet_method")


    def plot_03_08_dmdt_ice_sheet_method(self):
        """
        plot 2003-2008 dM/dt x ice sheet x method
        -----

        """
        r = ['GrIS', 'APIS', 'EAIS', 'WAIS', 'AIS', 'AIS+GrIS']
        m = ['IOM', 'RA', 'ICESat', 'GRACE']
        dmdt = {
            'GrIS': {'IOM': -284, 'RA': np.NAN, 'ICESat': -185, 'GRACE': -227, 'All': -232},
            'APIS': {'IOM': -36, 'RA': np.NAN, 'ICESat': -28, 'GRACE': -21, 'All': -28},
            'EAIS': {'IOM': -30, 'RA': 22, 'ICESat': 109, 'GRACE': 35, 'All': 24},
            'WAIS': {'IOM': -77, 'RA': -54, 'ICESat': -60, 'GRACE': -68, 'All': -67},
            'AIS': {'IOM': -142, 'RA': np.NAN, 'ICESat': 21, 'GRACE': -57, 'All': -72},
            'AIS+GrIS': {'IOM': -427, 'RA': np.NAN, 'ICESat': -164, 'GRACE': -284, 'All': -304},
        }
        dmdt_sd = {
            'GrIS': {'IOM': 65, 'RA': np.NAN, 'ICESat': 24, 'GRACE': 30, 'All': 23},
            'APIS': {'IOM': 17, 'RA': np.NAN, 'ICESat': 18, 'GRACE': 14, 'All': 10},
            'EAIS': {'IOM': 76, 'RA': 39, 'ICESat': 57, 'GRACE': 40, 'All': 36},
            'WAIS': {'IOM': 38, 'RA': 27, 'ICESat': 39, 'GRACE': 23, 'All': 21},
            'AIS': {'IOM': 86, 'RA': np.NAN, 'ICESat': 81, 'GRACE': 50, 'All': 43},
            'AIS+GrIS': {'IOM': 108, 'RA': np.NAN, 'ICESat': 84, 'GRACE': 58, 'All': 49},
        }
        self.plot_dmdt_ice_sheet_method(dmdt, dmdt_sd, r, m)

    def plot_dmdt_sheet(self, ice_name, t_sats, dmdt_sats, dmdt_sd_sats, order=None, subplot=None):
        """
        ====================================================================
        |            plot APIS WAIS EAIS GrIS dmdt v t x method            |
        ====================================================================
        Plots the rate of change in mass (Gt/yr) of an ice sheet, showing
        separate plots for each satellite.

        INPUTS:
            ice_name:       The name of the ice sheet
            t_sats:         A dictionary containing the time-series of each
                            satellite's observations
            dmdt_sats:      A dict containing dm/dt per satellite
            dmdt_sd_sats:   A dict containing standard dev. of dmdt_sats
        """
        if subplot is None:
            _, ax1 = plt.subplots()
        else:
            ax1 = subplot
        # set axes limits
        ax1.set_xlim(1992, 2012)
        ax1.set_ylim(-475, 325)
        # set title & labels
        ax1.set_title(self.long_names.get(ice_name, ice_name))
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Mass Balance (Gt/yr)')

        # if no order has been specified, get names from
        #  the time dictionary (note, this causes random ordering)
        if order is None:
            order = t_sats.keys()

        patches = []
        for z, sat_name in enumerate(order):
            t = t_sats[sat_name]
            dmdt = dmdt_sats[sat_name]
            dmdt_sd = dmdt_sd_sats[sat_name]
            # get colours associated with series
            col_a, col_b = self.sat_cols[sat_name]
            # create the plot
            plot_err_poly(t, dmdt, dmdt_sd, ax=ax1, average=True, nsigma=3, col1=col_a, col2=col_b, zrank=z*2)

            # get the long name of the series
            legend_name = self.long_names.get(sat_name, sat_name)
            # and add it to the legend
            patches.append(mpatches.Patch(color=col_b, label=legend_name))

        # create the legend object
        l = ax1.legend(handles=patches, loc=3)
        # add colours to the text of the legend
        for text, patch in zip(l.get_texts(), l.get_patches()):
            col = patch.get_facecolor()
            text.set_color(col)

        # make sure the legend is on top.
        l.set_zorder((z + 1) * 2)

        self.output_plot("dmdt_ice_sheet_{}".format(ice_name.lower()))

    def plot_reconciled(self, t_sheets, dmdt_sheets, dmdt_sd_sheets, order=None, subplot=None):
        """
        ====================================================================
        |              plot APIS WAIS EAIS GrIS x reconciled               |
        ====================================================================
        Plots the rate of change in mass (Gt/yr) of each ice sheet, with
        a second axis to show the corresponding sea level contribution.

        INPUTS:
            t_sheets:       A dictionary containing the time-series of each
                            ice sheet's data
            dmdt_sheets:    A dict containing dm/dt per ice sheet
            dmdt_sd_sheets: A dict containing standard dev. of dmdt_sheets
        """
        if order is None:
            order = t_sheets.keys()

        if subplot is None:
            _, ax1 = plt.subplots()
        else:
            ax1 = subplot
        ax2 = ax1.twinx()

        ax1.set_xlim(1992, 2013)

        ax1.set_xlabel('Year')
        ax1.set_ylabel('Mass (Gt)')
        ax2.set_ylabel('Sea level contribution (mm)')
        ax2.grid(b=False)

        patches = []
        for z, sheet_name in enumerate(order):
            t = t_sheets[sheet_name]
            dmdt = dmdt_sheets[sheet_name]
            dmdt_sd = dmdt_sd_sheets[sheet_name]
            col_a, col_b = self.sheet_cols[sheet_name]

            plot_err_poly(t, dmdt, dmdt_sd, ax=ax1, average=True, nsigma=3, col1=col_a, col2=col_b, zrank=z*2)

            legend_name = self.long_names.get(sheet_name, sheet_name)
            patches.append(mpatches.Patch(color=col_b, label=legend_name))

        l = ax1.legend(handles=patches, loc=3)
        for text, patch in zip(l.get_texts(), l.get_patches()):
            col = patch.get_facecolor()
            text.set_color(col)

        l.set_zorder((z + 1) * 2)

        miny, maxy = ax1.get_ylim()
        ax2.set_ylim(miny/-360., maxy/-360.)
        ax1.minorticks_on()

        names = '_'.join(order).lower()
        self.output_plot("reconciled_mass_{}".format(names))

    def plot_ice_mass(self, ice_name, t_sats, mass_sats, order=None, subplot=None):
        """
        ====================================================================
        |              plot APIS WAIS EAIS GrIS m v t x method             |
        ====================================================================
        Plots the total change in mass (Gt) of an ice sheet, showing
        separate plots for each satellite.

        INPUTS:
            ice_name:       The name of the ice sheet
            t_sats:         A dictionary containing the time-series of each
                            satellite's observations
            mass_sats:      A dict containing mass per satellite
        """
        if order is None:
            order = t_sats.keys()

        if subplot is None:
            _, ax1 = plt.subplots()
        else:
            ax1 = subplot
        ax2 = ax1.twinx()

        ax1.set_xlim(1992, 2012)

        plt.title(self.long_names.get(ice_name, ice_name))
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Mass (Gt)')
        ax2.set_ylabel('Sea level contribution (mm)')
        ax2.grid(b=False)

        patches = []
        for sat_name in order:
            t = t_sats[sat_name]
            mass = mass_sats[sat_name]
            _, col = self.sat_cols[sat_name]

            ax1.plot(t, mass, color=col, linewidth=2)

            legend_name = self.long_names.get(sat_name, sat_name)
            patches.append(mpatches.Patch(color=col, label=legend_name))

        l = ax1.legend(handles=patches, loc=3)
        for text, patch in zip(l.get_texts(), l.get_patches()):
            col = patch.get_facecolor()
            text.set_color(col)

        miny, maxy = ax1.get_ylim()
        ax2.set_ylim(miny/-360., maxy/-360.)
        ax1.minorticks_on()

        self.output_plot("mass_ice_sheet_{}".format(ice_name.lower()))
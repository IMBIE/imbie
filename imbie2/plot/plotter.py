import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
import numpy as np
import pandas as pd
import math
import os
import importlib
import inspect
from typing import Sequence, Tuple

from . import plots
from . import style

from functools import wraps
from itertools import cycle, product
from cycler import cycler
from collections import OrderedDict

from imbie2.const.basins import *
from imbie2.const import AverageMethod
from imbie2.model.collections import MassChangeCollection, MassRateCollection, WorkingMassRateCollection
from imbie2.util.functions import ts2m, move_av, match, t2m
from imbie2.util.combine import weighted_combine as ts_combine
from imbie2.model.managers import MassChangeCollectionsManager, MassRateCollectionsManager
from imbie2.model.series import *
from imbie2.proc.compare_windows import WindowStats


def chunk_rates(series):
    ok = series.t0 == series.t1

    time_chunks = [series.t0[ok]]
    dmdt_chunks = [series.dmdt[ok]]
    errs_chunks = [series.errs[ok]]

    for i in range(len(series)):
        if ok[i]: continue

        time_chunks.append(
            np.asarray([series.t0[i], series.t1[i]])
        )
        dmdt_chunks.append(
            np.asarray([series.dmdt[i], series.dmdt[i]])
        )
        errs_chunks.append(
            np.asarray([series.errs[i], series.errs[i]])
        )

    t, dmdt = ts_combine(time_chunks, dmdt_chunks)
    _, errs = ts_combine(time_chunks, errs_chunks, error=True)

    return t, dmdt, errs


def apply_offset(t, mass, mid):
    i_mid = 0
    near = mid
    for i, it in enumerate(t):
        diff = abs(it - mid)
        if diff < near:
            i_mid = i
            near = diff

    offset = mass[i_mid]
    return mass - offset


def sum_sheets(ts, data):
    t, _ = ts_combine(ts, data)
    out = np.zeros(t.shape, dtype=np.float64)

    beg_t = np.min(ts[0])
    end_t = np.max(ts[0])

    for i, times in enumerate(ts):
        tm, dm = ts2m(times, data[i])
        i1, i2 = match(t, tm, 1e-8)
        out[i1] += dm[i2]

        min_tm = np.min(tm)
        max_tm = np.max(tm)
        if min_tm > beg_t:
            beg_t = min_tm
        if max_tm > end_t:
            end_t = max_tm

    ok = np.logical_and(
        t >= beg_t,
        t <= end_t
    )
    return t[ok], out[ok]


def render_plot(method):
    @wraps(method)
    def wrapped(obj, *args, **kwargs):
        obj.clear_plot()
        ret = method(obj, *args, **kwargs)
        obj.draw_plot(ret)

    return wrapped


def render_plot_with_legend(method):
    @wraps(method)
    def wrapped(obj: "Plotter", *args, **kwargs):
        obj.clear_plot()
        obj.clear_legend()
        ret, leg = method(obj, *args, **kwargs)
        if leg.pop('extra', False):
            leg = obj.draw_legend(**leg)
            obj.draw_plot(ret, extra=(leg,))
        else:
            obj.draw_legend(**leg)
            obj.draw_plot(ret)

    return wrapped


class Plotter:
    _time0 = 1990
    _time1 = 2020
    _dmdt0 = -500 # -900
    _dmdt1 = 200 # 300
    _dm0 = -9000
    _dm1 = 3000
    _set_limits = True

    _imbie1_ylim_dmdt = -450, 300
    _imbie1_ylim_dm = -5000, 1000

    _sheet_names = {
        IceSheet.wais: "West Antarctica",
        IceSheet.eais: "East Antarctica",
        IceSheet.apis: "Antarctic Peninsula",
        IceSheet.gris: "Greenland",
        IceSheet.ais: "Antarctica",
        IceSheet.all: "Antarctica & Greenland"
    }
    _group_names = {
        "RA": "Altimetry",
        "GMB": "Gravimetry",
        "IOM": "Input-Output Method", # "Mass Budget",
        "LA": "Laser Altimetry",
        "all": "All"
    }

    def __init__(self, filetype=None, path=None, limits: bool=None):
        self._ext = filetype
        if path is None:
            path = os.getcwd()
        self._path = os.path.expanduser(path)

        if limits is not None:
            self._set_limits = limits

        mpl.rc('lines', linewidth=2)
        mpl.rc('font', size=22)
        mpl.rc('axes', linewidth=2)
        mpl.rc('xtick.major', width=1, size=5)
        mpl.rc('xtick.minor', width=1, size=3)
        mpl.rc('ytick.major', width=1, size=5)
        mpl.rc('ytick.minor', width=1, size=3)

    def _get_subplot_shape(self, count: int) -> Tuple[int, int, int]:
        if count == 1:
            w = 1
            h = 1
        elif count == 2:
            w = 2
            h = 1
        elif count == 3:
            w = 3
            h = 1
        elif count == 4:
            w = 2
            h = 2
        elif count <= 6:
            w = 3
            h = 2
        else:
            raise ValueError("unsupported number of subplots: {}".format(count))

        code = h * 100 + w * 10
        return w, h, code

    def draw_plot(self, fname=None, extra=None):
        if fname is None:
            plt.close(self.fig)
            return
        elif self._ext is None:
            plt.show()
        else:
            if not os.path.exists(self._path):
                print("creating directory: {}".format(self._path))
                os.makedirs(self._path)
            fname = fname+'.'+self._ext
            fpath = os.path.join(self._path, fname)

            if extra is None:
                plt.savefig(fpath, dpi=192, bbox_inches='tight')
            else:
                plt.savefig(fpath, bbox_extra_artists=extra, dpi=192, bbox_inches='tight')
            self.ax.clear()
            self.fig.clear()

            print("saving plot:", fpath)
        
        mpl.rc('lines', linewidth=2)
        mpl.rc('font', size=22)

    def clear_plot(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(9, 16))
        self.ax = plt.gca()

    def draw_legend(self, **legend_opts):
        if (not self.glyphs) or (not self.labels):
            return
        if 'parent' in legend_opts:
            if legend_opts.pop('parent') == 'fig':
                self.fig.legend(self.glyphs, self.labels, **legend_opts)
                return
        return plt.legend(self.glyphs, self.labels, **legend_opts)

    def clear_legend(self):
        self.glyphs = []
        self.labels = []

    @staticmethod
    def marker_glyph(marker):
        return mlines.Line2D([], [], color='k', marker=marker)

    @staticmethod
    def group_glyph(group):
        colour = style.colours.primary[group]
        return Plotter.colour_glyph(colour, label=group)

    @staticmethod
    def colour_glyph(colour, label=None, **kwargs):
        return mpatches.Patch(color=colour, label=label, **kwargs)

    @render_plot_with_legend
    def sheets_time_bars(self, data, sheets, names, *groups, suffix: str=None):
        if not groups:
            groups = ["RA", "GMB", "IOM"]
        num_sheets = len(sheets)
        plt_dims = 100 + 10 * num_sheets

        self.labels = [self._group_names[g] for g in groups]
        self.glyphs = [self.group_glyph(g) for g in groups]

        prev = None
        yticklabels = None
        for i, sheet in enumerate(sheets):
            plt_loc = plt_dims + i + 1

            if prev is None:
                ax = plt.subplot(plt_loc)
                prev = ax
            else:
                ax = plt.subplot(plt_loc, sharey=prev)
                if yticklabels is None:
                    yticklabels = ax.get_yticklabels()
                else:
                    yticklabels += ax.get_yticklabels()
            
            label = chr(ord('a') + i)
            ax.text(
                .05, .95, label,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )

            min_t = {}
            max_t = {}
            group = {}
            bsets = {}

            for series in data.filter(basin_id=sheet):
                u = series.user
                g = series.user_group

                min_t[u] = series.min_time
                max_t[u] = series.max_time
                if u not in bsets:
                    bsets[u] = [series.basin_group]
                else:
                    bsets[u].append(series.basin_group)
                group[u] = g

            order = []
            width = []
            start = []
            color = []

            for j, u in enumerate(names):
                if u not in group:
                    continue
                g = group[u]
                c = style.colours.primary[g]

                t0 = min_t[u]
                t1 = max_t[u]

                order.append(j-.4)
                width.append(t1 - t0)
                start.append(t0)
                color.append(c)

            ax.barh(order, width, height=.8, left=start, color=color)
            ax.set_xlim(self._time0, self._time1)
            ax.set_title(self._sheet_names[sheet])
            ax.set_xticks([1995, 2005, 2015])

        prev.set_yticks([i for i, _ in enumerate(names)])
        prev.set_yticklabels(names)
        if yticklabels is not None:
            plt.setp(yticklabels, visible=False)

        leg_params = {'loc': 'lower center', 'ncol': 3, 'parent': 'fig'}
        width = max(6, 4*len(sheets))
        self.fig.set_size_inches(width, 9)
        sheet_names = "_".join(s.value for s in sheets)
        
        mpl.rc('font', size=12)
        if suffix is not None:
            sheet_names += "_" + suffix
        return "sheets_time_bars_"+sheet_names, leg_params

    @render_plot_with_legend
    def split_time_bars(self, rate_data, mass_data, sheet, names, groups=None,
                        prefix=None):
        if groups is None:
            groups = ["RA", "GMB", "IOM"]
        if prefix is None: prefix = ''

        min_t = {BasinGroup.rignot: {'r': {}, 'm': {}},
                 BasinGroup.zwally: {'r': {}, 'm': {}}}
        max_t = {BasinGroup.rignot: {'r': {}, 'm': {}},
                 BasinGroup.zwally: {'r': {}, 'm': {}}}
        group = {}
        plt_dims = 140

        self.labels = []
        self.glyphs = []

        for cat in BasinGroup.rignot, BasinGroup.zwally:
            for rate in rate_data[sheet]:
                if rate.computed: continue
                if rate.basin_group != cat:
                    continue

                g = rate.user_group
                u = rate.user
                group[u] = g

                min_t[cat]['r'][u] = rate.min_time
                max_t[cat]['r'][u] = rate.max_time

                label = self._group_names[g]
                if label not in self.labels:
                    self.labels.append(label)
                    self.glyphs.append(self.group_glyph(g))
            for mass in mass_data[sheet]:
                if mass.computed: continue
                if mass.basin_group != cat:
                    continue

                g = mass.user_group
                u = mass.user
                group[u] = g

                min_t[cat]['m'][u] = mass.min_time
                max_t[cat]['m'][u] = mass.max_time

        prev = None
        yticklabels = None
        for i, cat in enumerate([BasinGroup.rignot, BasinGroup.zwally]):
            for k, d in enumerate(['r', 'm']):
                plt_loc = plt_dims + i + 1 + k * 2
                if prev is None:
                    ax = plt.subplot(plt_loc)
                    prev = ax
                else:
                    ax = plt.subplot(plt_loc, sharey=prev)
                    if yticklabels is None:
                        yticklabels = ax.get_yticklabels()
                    else:
                        yticklabels += ax.get_yticklabels()

                order = []
                width = []
                start = []
                color = []

                for j, u in enumerate(names):
                    if u not in min_t[cat][d]:
                        continue
                    g = group[u]

                    t0 = min_t[cat][d][u]
                    t1 = max_t[cat][d][u]
                    c = style.colours.primary[g]

                    order.append(j-.4)
                    width.append(t1 - t0)
                    start.append(t0)
                    color.append(c)

                ax.barh(order, width, height=.8, left=start, color=color)
                ax.set_xlim(self._time0, self._time1)
                _type = {'r': "dM/dt", 'm': "dM"}[d]
                ax.set_title(cat.value+' ('+_type+')')
        self.fig.suptitle(self._sheet_names[sheet])
        prev.set_yticks([i for i, _ in enumerate(names)])
        prev.set_yticklabels(names)
        plt.setp(yticklabels, visible=False)

        self.fig.set_size_inches(16, 9)
        return prefix+"_split_time_bars_"+sheet.value, {'loc': 4, 'frameon': False}


    @render_plot
    def rignot_zwally_comparison(self, data: WorkingMassRateCollection, sheets: Sequence[IceSheet]) -> str:

        for sheet in sheets:
            rignot_data = data.filter(
                basin_group=BasinGroup.rignot, basin_id=sheet
            )
            zwally_data = data.filter(
                basin_group=BasinGroup.zwally, basin_id=sheet
            )
            zwally_users = {s.user for s in zwally_data}
            rignot_users = {s.user for s in rignot_data}
            users = zwally_users.intersection(rignot_users)

            for user in users:
                zwally_series = zwally_data.filter(user=user).first()
                rignot_series = rignot_data.filter(user=user).first()

                self.ax.errorbar(
                    x=zwally_series.mean, xerr=zwally_series.sigma,
                    y=rignot_series.mean, yerr=rignot_series.sigma,
                    ecolor=style.primary[zwally_series.user_group]
                )

        ymin, ymax = self.ax.get_ylim()
        xmin, xmax = self.ax.get_xlim()
        min_rate = min(xmin, ymin)
        max_rate = max(xmax, ymax)

        self.ax.plot(
            [min_rate, max_rate], [min_rate, max_rate], 'k--'
        )

        self.ax.set_xlim(min_rate, max_rate)
        self.ax.set_ylim(min_rate, max_rate)
        self.ax.set_xlabel('Zwally dM/dt (Gt/yr)')
        self.ax.set_ylabel('Rignot dM/dt (Gt/yr)')
        self.fig.suptitle("Rignot/Zwally Comparison")
        if len(sheets) == 1:
            self.ax.set_title(self._sheet_names[sheets[0]])
        self.ax.grid()

        return "rignot_zwally_comparison_" + "_".join([s.value for s in sheets])

    @render_plot_with_legend
    def basin_errors(self, basins, data, name, sheets=None):
        min_y = 0
        max_y = 0
        max_z = 0

        x = 0
        names = []
        groups = set()
        sheet_ranges = {}

        if sheets is None:
            sheets = [s for s in IceSheet if s is not IceSheet.ais]

        for sheet in sheets:

            n_basins = 0
            min_x = x

            for basin in basins.sheet(sheet):
                n_basins += 1
                names.append(basin.value)

                z = 0
                w = .8

                items = OrderedDict()
                for record in data[basin]:
                    if record.user_group not in items:
                        items[record.user_group] = {
                            'means': [],
                            'sigmas': []
                        }
                    items[record.user_group]['means'].append(record.mean)
                    items[record.user_group]['sigmas'].append(record.sigma)

                for group in items:
                    groups.add(group)

                    mean = np.mean(items[group]['means'])
                    sigma = math.sqrt(
                        np.mean(np.square(items[group]['sigmas']))
                    ) # / math.sqrt(len(items))

                    col_a = style.colours.primary[group]
                    col_b = style.colours.secondary[group]

                    margin = 1. - (w / 2)
                    xa = x + margin

                    ya = mean - sigma * 2
                    h = sigma * 4

                    min_y = min(min_y, ya)
                    max_y = max(max_y, ya+h)

                    rect = mpatches.Rectangle(
                        (xa, ya), w, h, zorder=z, color=col_b
                    )
                    self.ax.add_patch(rect)

                    z += 1
                    ya = mean - sigma
                    h = sigma * 2

                    rect = mpatches.Rectangle(
                        (xa, ya), w, h, zorder=z, color=col_a
                    )
                    self.ax.add_patch(rect)

                    z += 1
                    ya = mean
                    self.ax.plot(
                        [xa, xa+w], [ya, ya],
                        color='white', zorder=z,
                        linewidth=2
                    )

                    w -= .2
                x += 1
                max_z = max(max_z, z)
            sheet_ranges[sheet] = (min_x + .8, x + .2)
            self.ax.axvline(x+.5, color='k', ls='--')

        text_y = max_y - 4.5
        line_y = max_y - 5

        for sheet in sheet_ranges:
            xa, xb = sheet_ranges[sheet]

            xc = xa + (xb - xa) / 2
            self.ax.text(
                xc, text_y, sheet.value.upper(),
                horizontalalignment='center',
                zorder=max_z+1
            )
            self.ax.plot(
                [xa, xb], [line_y, line_y],
                'k-', linewidth=2, zorder=max_z+1
            )

        self.labels = [self._group_names[g] for g in groups]
        self.glyphs = [self.group_glyph(g) for g in groups]
        # draw line at 0 dM/dt
        self.ax.axhline(0, color='black')
        # set x and y ranges
        self.ax.set_xlim(.5, x+.5)
        self.ax.set_ylim(min_y, max_y)
        # move x ticks to top
        self.ax.xaxis.tick_top()
        # turn on minor ticks (both axes), then disable x-axis minor ticks
        self.ax.minorticks_on()
        self.ax.tick_params(axis='x', which='minor', top='off')
        # set position and labels for x-axis major ticks
        self.ax.set_xticks(np.arange(1, x+1))
        self.ax.set_xticklabels(names)

        plt.ylabel("Rate of Mass Change (Gt/yr)")
        plt.title("Rate of Mass Change per {} Drainage Basin\n".format(name))
        self.fig.set_size_inches(16, 9)

        # return plot name
        return "errors_" + name.lower(), {"frameon": False, "loc": 4}

    @render_plot_with_legend
    def sheets_error_bars(self, group_avgs: WorkingMassRateCollection, sheet_avgs: WorkingMassRateCollection,
                          methods: Sequence[str], sheets: Sequence[IceSheet], ylabels: bool=False,
                          window: Tuple[float, float]=None, suffix: str=None):
        # get mean & error dM/dt per ice-sheet and group
        width = len(methods)
        min_y = None
        max_y = None
        max_x = 0

        if window is None:
            t_min, t_max = None, None
        else:
            t_min, t_max = window

        for i, sheet in enumerate(sheets):
            max_x += width+1
            # plot all-group patches
            sheet_series = sheet_avgs.filter(
                basin_id=sheet
            ).first().truncate(t_min, t_max)

            if sheet_series is None:
                continue

            mean = sheet_series.mean
            err = sheet_series.sigma

            y = mean - err
            x = i * (width + 1)

            rect = mpatches.Rectangle(
                (x, y), width, err*2, color=style.colours.secondary['all']
            )
            self.ax.add_patch(rect)

            if min_y is None or y - err < min_y:
                min_y = y - err
            if max_y is None or y + err > max_y:
                max_y = y + err
            # plot error bars
            for j, method in enumerate(methods):
                group_series = group_avgs.filter(
                    user_group=method, basin_id=sheet
                ).first().truncate(t_min, t_max)

                if group_series is None:
                    continue

                mean = group_series.mean
                err = group_series.sigma

                jx = x + j + .5
                y1 = mean - err
                y2 = mean + err

                self.ax.errorbar(
                    [jx], [mean], yerr=[err],
                    color=style.colours.primary[method],
                    linewidth=10, capsize=0
                )
                if y1 < min_y:
                    min_y = y1
                if y2 > max_y:
                    max_y = y2
        # add legend
        for group in methods:
            self.glyphs.append(
                self.group_glyph(group)
            )
            self.labels.append(
                self._group_names[group]
            )

        # set axis labels, limits, ticks
        names = [self._sheet_names[s] for s in sheets]
        ticks = [(i+.5)*width+i for i, _ in enumerate(sheets)]

        y_margin = (max_y - min_y) * .05

        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(names)
        self.ax.set_xlim(-.5, max_x-.5)
        self.ax.set_ylim(min_y - y_margin, max_y + y_margin)

        if not ylabels:
            self.ax.tick_params(
                axis='y', which='both', left='off', labelleft='off', right='off')
            ax2 = None
        else:
            plt.ylabel("dM/dt (Gt $\mathregular{yr^{-1}}$)")
            ax2 = self.ax.twinx()
            _min, _max = self.ax.get_ylim()
            lims = _min / -360, _max / -360
            ax2.set_ylim(lims)
            ax2.set_ylabel('Sea Level Contribution (mm $\mathregular{yr^{-1}}$)')

        plt.xlabel("Ice Sheet", fontweight='bold')
        width = max(6, 4*len(sheets))
        self.fig.set_size_inches(width, 9)

        name = "sheets_error_bars"
        if suffix is not None:
            name += "_" + suffix

        if len(sheets) == 1:
            leg_style = dict(
                loc='lower center', parent='fig', frameon=False
            )
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0+box.height*.25, box.width, box.height*.75])
            if ax2 is not None:
                ax2.set_position([box.x0, box.y0+box.height*.25, box.width, box.height*.75])
        else:
            leg_style = dict(frameon=False)
        return name, leg_style

    @render_plot
    def coverage_combined(self, stack_data, coverage_data, names):
        """
        """
        self.fig, axs = plt.subplots(nrows=2)
        self.fig.set_size_inches(9, 32)

        # named coverage plot
        groups = ["RA", "GMB", "IOM"]
        self.labels = [self._group_names[g] for g in groups]
        self.glyphs = [self.group_glyph(g) for g in groups]
            
        axs[0].text(
            .05, .95, 'a',
            horizontalalignment='left',
            verticalalignment='top',
            transform=axs[0].transAxes
        )

        min_t = {}
        max_t = {}
        group = {}
        bsets = {}

        for series in coverage_data:
            u = series.user
            g = series.user_group

            min_t[u] = series.min_time
            max_t[u] = series.max_time
            if u not in bsets:
                bsets[u] = [series.basin_group]
            else:
                bsets[u].append(series.basin_group)
            group[u] = g

            order = []
            width = []
            start = []
            color = []

        for j, u in enumerate(names):
            if u not in group:
                continue
            g = group[u]
            c = style.colours.primary[g]

            t0 = min_t[u]
            t1 = max_t[u]

            order.append(j-.4)
            width.append(t1 - t0)
            start.append(t0)
            color.append(c)

        axs[0].barh(order, width, height=.8, left=start, color=color)
        axs[0].set_xlim(self._time0, self._time1)
        axs[0].set_xticks([1995, 2005, 2015])

        axs[0].set_yticks([i for i, _ in enumerate(names)])
        axs[0].set_yticklabels(names)
        axs[1].text(
            .05, .95, 'b',
            horizontalalignment='left',
            verticalalignment='top',
            transform=axs[1].transAxes
        )

        order = ['IOM', 'RA', 'GMB']
        users = {series.user for series in stack_data}
        xs = []
        for user in users:
            user_data = stack_data.filter(user=user)
            min_time = min(series.min_time for series in user_data)
            max_time = max(series.max_time for series in user_data)
            xs.extend([min_time-.01, min_time+.01,
                       max_time-.01, max_time+.01])
        xs.sort()
        xs = np.asarray(xs, dtype=float)

        counts = {}
        for g in order:
            counts[g] = np.zeros(xs.shape, dtype=int)

        for user in users:
            user_data = stack_data.filter(user=user)
            min_time = min(series.min_time for series in user_data)
            max_time = max(series.max_time for series in user_data)

            ok = np.logical_and(
                xs > min_time,
                xs < max_time
            )
            counts[user_data.first().user_group][ok] += 1

        for g in order:
            self.labels.append(
                self._group_names[g]
            )
            self.glyphs.append(
                self.group_glyph(g)
            )
        axs[1].stackplot(
            xs, [counts[g] for g in order],
            colors=[style.primary[g] for g in order],
            alpha=.5
        )

        return "coverage_combined"

    @render_plot
    def windows_comparison(self, data: Sequence[WindowStats], suffix: str=None):
        """
        stacked bar plot of number of submissions per time window
        """
        prev = [0 for _ in data]
        xs = np.arange(len(data))
        names = [
            "{}-\n{}".format(s.start, s.end) for s in data
        ]

        for group in "RA", "GMB", "IOM":
            counts = [s.groups[group] for s in data]
            self.ax.bar(
                xs, counts, .35, bottom=prev,
                color=style.primary[group]
            )
            prev = [p+c for p, c in zip(prev, counts)]

        self.ax.set_xticks(xs)
        self.ax.set_xticklabels(names)
        self.ax.set_title("Data coverage per window")
        self.ax.set_xlabel("window")
        self.ax.set_ylabel("contributions")

        if suffix is None:
            suffix = ""
        else:
            suffix = "_" + suffix
        return "windows_comparison" + suffix

    @render_plot_with_legend
    def stacked_coverage(self, data: Union[WorkingMassRateCollection, MassChangeCollection], suffix: str=None):
        """
        stacked area plot of number of submissions over time
        """
        xs = []
        users = {series.user for series in data}
        for user in users:
            user_data = data.filter(user=user)
            min_time = min(series.min_time for series in user_data)
            max_time = max(series.max_time for series in user_data)
            xs.extend([min_time-.01, min_time+.01,
                       max_time-.01, max_time+.01])
        xs.sort()
        xs = np.asarray(xs, dtype=float)

        counts = {}
        for g in 'GMB', 'RA', 'IOM':
            counts[g] = np.zeros(xs.shape, dtype=int)
        for user in users:
            user_data = data.filter(user=user)
            min_time = min(series.min_time for series in user_data)
            max_time = max(series.max_time for series in user_data)

            ok = np.logical_and(
                xs > min_time,
                xs < max_time
            )
            counts[user_data.first().user_group][ok] += 1

        for g in 'IOM', 'RA', 'GMB':
            self.labels.append(
                self._group_names[g]
            )
            self.glyphs.append(
                self.group_glyph(g)
            )
        self.ax.stackplot(xs, [counts[g] for g in ['IOM', 'RA', 'GMB']],
                          colors=[style.primary[g] for g in ['IOM', 'RA', 'GMB']],
                          alpha=.5)

        name = "stacked_coverage"
        if suffix is not None:
            name += "_"+suffix

        return name, dict(loc='top left', frameon=False, framealpha=0)

    @render_plot
    def group_rate_boxes(self, rate_data: WorkingMassRateCollection, regions, suffix: str=None):
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)

            for series in rate_data.filter(basin_id=sheets):
                pcol = style.colours.primary[series.user_group]

                t_min = series.min_time
                t_ext = series.max_time - t_min

                r_min = series.min_rate
                r_ext = series.max_rate - r_min

                if r_ext == 0:
                    r_min -= series.errs[0]
                    r_ext += series.errs[0]

                if series.computed:
                    rect = mpatches.Rectangle(
                        (t_min, r_min),
                        t_ext, r_ext,
                        edgecolor=pcol, hatch='\\/',
                        fill=None
                    )
                else:
                    rect = mpatches.Rectangle(
                        (t_min, r_min),
                        t_ext, r_ext,
                        facecolor=pcol, alpha=.4
                    )
                self.ax.add_patch(rect)

            # set title & axis labels
            self.ax.set_ylabel("Mass Balance (Gt/yr)")
            self.ax.set_title(self._sheet_names[name])
            # set x- & y-axis limits
            if self._set_limits:
                self.ax.set_ylim(self._dmdt0, self._dmdt1)
                self.ax.set_xlim(self._time0, self._time1)

            show_yaxis = i % plt_w == 0
            show_xaxis = int(i / plt_w) == plt_h - 1

            self.ax.get_yaxis().set_visible(show_yaxis)
            self.ax.get_xaxis().set_visible(show_xaxis)

        self.fig.suptitle("<dM/dt> Data Coverage")
        self.fig.autofmt_xdate()

        name = "group_rate_boxes"
        name += "_".join([s.value for s in regions])
        if suffix is not None:
            name += "_" + suffix
        return name

    @render_plot_with_legend
    def sheet_scatter(self, ice_sheet: IceSheet, basins: BasinGroup, data: WorkingMassRateCollection):
        markers = cycle(style.markers)
        groups = set()
        legend_style = {
            'frameon': False,
            'prop': {'size': 8}
        }

        for marker, basin in zip(markers, basins.sheet(ice_sheet)):
            basin_data = data.filter(basin_id=basin)

            if not basin_data:
                continue

            _new = plots.basin_scatter(self.ax, basin_data, marker)
            groups.update(_new)

            if _new:
                glyph = self.marker_glyph(marker)
                self.glyphs.append(glyph)
                self.labels.append(basin.value)

        self.ax.grid()
        self.ax.axhline(0, color='black')
        self.ax.set_xlim(2000, 2020)

        for group in groups:
            glyph = self.group_glyph(group)
            self.labels.append(group)
            self.glyphs.append(glyph)

        name = "{}: {}".format(
            ice_sheet.value, basins
        )
        plt.title(name)

        if groups:
            return "scatter_"+ice_sheet.value, legend_style
        return None, {}

    @render_plot
    def group_rate_intracomparison(self, group_avgs: WorkingMassRateCollection,
            group_contribs: WorkingMassRateCollection, regions, suffix: str=None, mark: Sequence[str]=None) -> str:
        """
        dM/dt comparison within a group
        """
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))
        if mark is None:
            mark = []

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)
            self.ax.axhline(0, ls='--', color='k')

            avg = group_avgs.filter(basin_id=name).average()
            if avg is None:
                print(name)
                continue

            pcol = style.colours.primary[avg.user_group]
            scol = style.colours.secondary[avg.user_group]

            self.ax.plot(avg.t, avg.dmdt, color=pcol)
            self.ax.fill_between(
                avg.t, avg.dmdt - avg.errs, avg.dmdt + avg.errs,
                color=scol, alpha=.5
            )

            if len(sheets) == 1:
                for contrib in group_contribs.filter(basin_id=sheets):
                    self.ax.plot(contrib.t, contrib.dmdt, color=pcol, ls='--')
                    if contrib.user in mark:
                        mid = len(contrib.t) // 2
                        x = contrib.t[mid]
                        y = contrib.dmdt[mid]
                        self.ax.annotate(
                            contrib.user,
                            xy=(x, y), xytext=(-20, 10),
                            textcoords='offset points', ha='left', va='bottom',
                            arrowprops=dict(arrowstyle='->'))

            # get start & end time of common period
            com_t_min = group_contribs.filter(basin_id=sheets).concurrent_start()
            com_t_max = group_contribs.filter(basin_id=sheets).concurrent_stop()
            # plot v. lines to show period
            self.ax.axvline(com_t_min, ls='--', color='k')
            self.ax.axvline(com_t_max, ls='--', color='k')

            # set title & axis labels
            self.ax.set_ylabel("Mass Balance (Gt/yr)")
            self.ax.set_title(self._sheet_names[name])
            # set x- & y-axis limits
            if self._set_limits:
                self.ax.set_ylim(self._dmdt0, self._dmdt1)
                self.ax.set_xlim(self._time0, self._time1)

            show_yaxis = i % plt_w == 0
            show_xaxis = int(i / plt_w) == plt_h - 1

            self.ax.get_yaxis().set_visible(show_yaxis)
            self.ax.get_xaxis().set_visible(show_xaxis)

        self.fig.suptitle("dM/dt intracomparison")
        self.fig.autofmt_xdate()

        name = "group_rate_intracomparison_"
        name += "_".join([s.value for s in regions])
        if suffix is not None:
            name += "_" + suffix
        return name

    @render_plot
    def group_mass_intracomparison(self, group_avgs: MassChangeCollection, group_contribs: MassChangeCollection,
                                   regions, suffix: str=None, mark: Sequence[str]=None, align: bool=False) -> str:
        """
        dM comparison within a group
        """
        if mark is None:
            mark = []
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)
            self.ax.axhline(0, ls='--', color='k')

            avg = group_avgs.filter(basin_id=name).average()
            if avg is None:
                print(name)
                continue

            pcol = style.colours.primary[avg.user_group]
            scol = style.colours.secondary[avg.user_group]

            self.ax.plot(avg.t, avg.mass, color=pcol)
            self.ax.fill_between(
                avg.t, avg.mass - avg.errs, avg.mass + avg.errs,
                color=scol, alpha=.5
            )

            if len(sheets) == 1:
                for contrib in group_contribs.filter(basin_id=sheets):
                    if align:
                        contrib = contrib.align(avg)
                    self.ax.plot(contrib.t, contrib.mass, color=pcol, ls='--')
                    if contrib.user in mark:
                        x = contrib.t[-1]
                        y = contrib.mass[-1]
                        self.ax.annotate(
                            contrib.user,
                            xy=(x, y), xytext=(-20, 10),
                            textcoords='offset points', ha='left', va='bottom',
                            arrowprops=dict(arrowstyle='->'))

            # get start & end time of common period
            com_t_min = group_contribs.filter(basin_id=sheets).concurrent_start()
            com_t_max = group_contribs.filter(basin_id=sheets).concurrent_stop()
            # plot v. lines to show period
            self.ax.axvline(com_t_min, ls='--', color='k')
            self.ax.axvline(com_t_max, ls='--', color='k')

            # set title & axis labels
            self.ax.set_ylabel("Mass Change (Gt)")
            self.ax.set_title(self._sheet_names[name])
            # set x- & y-axis limits
            if self._set_limits:
                self.ax.set_ylim(self._dm0, self._dm1)
                self.ax.set_xlim(self._time0, self._time1)

            show_yaxis = i % plt_w == 0
            show_xaxis = int(i / plt_w) == plt_h - 1

            self.ax.get_yaxis().set_visible(show_yaxis)
            self.ax.get_xaxis().set_visible(show_xaxis)

        self.fig.suptitle("dM intracomparison")
        self.fig.autofmt_xdate()

        name = "group_mass_intracomparison_"
        name += "_".join([s.value for s in regions])
        if suffix is not None:
            name += "_" + suffix
        return name

    @render_plot
    def groups_rate_intercomparison(self, region_avgs: WorkingMassRateCollection, group_avgs: WorkingMassRateCollection,
                                    regions, groups=None) -> str:
        """
        dM/dt comparison between groups
        """
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))
        self.fig.set_size_inches(16, 9)

        if groups is None:
            groups = ["IOM", "RA", "GMB"]

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)
            self.ax.axhline(0, ls='--', color='k')

            x_avg = region_avgs.filter(basin_id=name).average()
            if x_avg is None:
                print(name)
                continue

            pcol = style.colours.primary["all"]
            scol = style.colours.secondary["all"]

            self.ax.fill_between(
                x_avg.t, x_avg.dmdt - x_avg.errs, x_avg.dmdt + x_avg.errs,
                color=scol
            )
            self.ax.plot(x_avg.t, x_avg.dmdt, color=pcol)

            for g in groups:
                g_avg = group_avgs.filter(basin_id=name, user_group=g).first()

                pcol = style.colours.primary[g_avg.user_group]
                scol = style.colours.secondary[g_avg.user_group]

                self.ax.fill_between(
                    g_avg.t, g_avg.dmdt - g_avg.errs, g_avg.dmdt + g_avg.errs,
                    color=scol, alpha=.75
                )
                self.ax.plot(g_avg.t, g_avg.dmdt, color=pcol)

            # get start & end time of common period
            com_t_min = group_avgs.concurrent_start()
            com_t_max = group_avgs.concurrent_stop()
            # plot v. lines to show period
            self.ax.axvline(com_t_min, ls=':', color='#888888')
            self.ax.axvline(com_t_max, ls=':', color='#888888')

            # set title & axis labels
            self.ax.set_ylabel("Mass Balance (Gt/yr)")
            self.ax.set_title(self._sheet_names[name])
            # set x- & y-axis limits
            if self._set_limits:
                self.ax.set_ylim(self._dmdt0, self._dmdt1)
                self.ax.set_xlim(self._time0, self._time1)

            show_yaxis = i % plt_w == 0
            show_xaxis = int(i / plt_w) == plt_h - 1

            self.ax.get_yaxis().set_visible(show_yaxis)
            self.ax.get_xaxis().set_visible(show_xaxis)

        self.fig.suptitle("dM/dt intercomparison")
        self.fig.autofmt_xdate()

        name = "groups_rate_intercomparison_"
        name += "_".join([s.value for s in regions])
        return name

    @render_plot
    def groups_mass_intercomparison(self, region_avgs: MassChangeCollection, group_avgs: MassChangeCollection,
                                    regions, align: bool=False, groups=None) -> str:
        """
        dM comparison between groups
        """
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))
        if groups is None:
            groups = ["IOM", "RA", "GMB"]

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)
            self.ax.axhline(0, ls='--', color='k')

            x_avg = region_avgs.filter(basin_id=name).first()
            if x_avg is None:
                print(name)
                continue

            pcol = style.colours.primary["all"]
            scol = style.colours.secondary["all"]

            self.ax.fill_between(
                x_avg.t, x_avg.mass - x_avg.errs, x_avg.mass + x_avg.errs,
                color=scol
            )
            self.ax.plot(x_avg.t, x_avg.mass, color=pcol)

            for g in groups:
                g_avg = group_avgs.filter(basin_id=name, user_group=g).first()
                if align:
                    g_avg = g_avg.align(x_avg)
                pcol = style.colours.primary[g_avg.user_group]
                scol = style.colours.secondary[g_avg.user_group]

                self.ax.fill_between(
                    g_avg.t, g_avg.mass - g_avg.errs, g_avg.mass + g_avg.errs,
                    color=scol, alpha=.75
                )
                self.ax.plot(g_avg.t, g_avg.mass, color=pcol)

            # get start & end time of common period
            com_t_min = group_avgs.concurrent_start()
            com_t_max = group_avgs.concurrent_stop()
            # plot v. lines to show period
            self.ax.axvline(com_t_min, ls='--', color='k')
            self.ax.axvline(com_t_max, ls='--', color='k')

            # set title & axis labels
            self.ax.set_title(self._sheet_names[name])
            # set x- & y-axis limits
            if self._set_limits:
                self.ax.set_ylim(self._dm0, self._dm1)
                self.ax.set_xlim(self._time0, self._time1)

            show_yaxis = i % plt_w == 0
            show_xaxis = int(i / plt_w) == plt_h - 1

            self.ax.get_yaxis().set_visible(show_yaxis)
            self.ax.get_xaxis().set_visible(show_xaxis)

        self.fig.suptitle("dM intercomparison")
        self.fig.autofmt_xdate()

        name = "groups_mass_intercomparison_"
        name += "_".join([s.value for s in regions])
        return name

    @render_plot_with_legend
    def regions_mass_intercomparison(self, region_avgs: MassChangeCollection, *regions: Sequence[IceSheet]) -> str:
        pcols = cycle(["#531A59", "#1B8C6F", "#594508", "#650D1B"])
        scols = cycle(["#9E58A5", "#4CA58F", "#D8B54D", "#A8152E"])
        self.ax.axhline(0, ls='--', color='k')

        for region, pcol, scol in zip(regions, pcols, scols):
            avg = region_avgs.filter(basin_id=region).first()

            self.ax.plot(avg.t, avg.mass, color=pcol)
            self.ax.fill_between(
                avg.t, avg.mass-avg.errs, avg.mass+avg.errs,
                color=scol, alpha=.5
            )
            self.labels.append(
                self._sheet_names[region]
            )
            self.glyphs.append(
                self.colour_glyph(pcol)
            )

        # get start & end time of common period
        com_t_min = region_avgs.concurrent_start()
        com_t_max = region_avgs.concurrent_stop()
        # plot v. lines to show period
        self.ax.axvline(com_t_min, ls='--', color='k')
        self.ax.axvline(com_t_max, ls='--', color='k')

        # set title & axis labels
        self.ax.set_ylabel("Mass Change (Gt)")
        self.ax.set_title("Intercomparison of Regions")
        self.fig.autofmt_xdate()
        # set x- & y-axis limits
        if self._set_limits:
            self.ax.set_ylim(self._dm0, self._dm1)
            self.ax.set_xlim(self._time0, self._time1)

        return "regions_mass_intercomparison_"+"_".join(r.value for r in regions), {"frameon": False, "loc": 3}

    @render_plot_with_legend
    def regions_rate_intercomparison(self, region_avgs: WorkingMassRateCollection, *regions: Sequence[IceSheet]) -> str:
        pcols = cycle(["#531A59", "#1B8C6F", "#594508", "#650D1B"])
        scols = cycle(["#9E58A5", "#4CA58F", "#D8B54D", "#A8152E"])
        self.ax.axhline(0, ls='--', color='k')

        for region, pcol, scol in zip(regions, pcols, scols):
            avg = region_avgs.filter(basin_id=region).average()

            self.ax.plot(avg.t, avg.dmdt, color=pcol)
            self.ax.fill_between(
                    avg.t, avg.dmdt - avg.errs, avg.dmdt + avg.errs,
                    color=scol, alpha=.5
            )
            self.labels.append(
                    self._sheet_names[region]
            )
            self.glyphs.append(
                    self.colour_glyph(pcol)
            )

        # get start & end time of common period
        com_t_min = region_avgs.concurrent_start()
        com_t_max = region_avgs.concurrent_stop()
        # plot v. lines to show period
        self.ax.axvline(com_t_min, ls='--', color='k')
        self.ax.axvline(com_t_max, ls='--', color='k')

        # set title & axis labels
        self.ax.set_ylabel("Rate of Mass Change (Gt/yr)")
        self.ax.set_title("Intercomparison of Regions")
        self.fig.autofmt_xdate()
        # set x- & y-axis limits
        if self._set_limits:
            self.ax.set_ylim(self._dmdt0, self._dmdt1)
            self.ax.set_xlim(self._time0, self._time1)

        return "regions_rate_intercomparison_" + "_".join(r.value for r in regions), {"frameon": False, "loc": 3}

    @render_plot_with_legend
    def named_dmdt_group_plot(self, region: IceSheet, group: str, data: WorkingMassRateCollection,
                              avg: WorkingMassRateDataSeries=None, full_dmdt: WorkingMassRateCollection=None,
                              alternative_avg: WorkingMassRateDataSeries=None,
                              colors: style.UsersColorCollection=None):
        data = data.filter(user_group=group, basin_id=region)
        self.ax.axhline(0, ls='--', color='k')

        if colors is None:
            users = list({s.user for s in data})
            colors = style.UsersColorCollection(users)

        glyphs = []
        labels = []
        min_errors = []
        max_errors = []

        if avg is not None:
            p = self.ax.plot(avg.t, avg.dmdt, label='Average', color='grey', lw=7)
            glyphs.append(p[0])
            labels.append('Average')
            min_err = "{:.2f}".format(np.nanmin(avg.errs))
            max_err = "{:.2f}".format(np.nanmax(avg.errs))
            min_errors.append(min_err)
            max_errors.append(max_err)

            self.ax.fill_between(
                avg.t, avg.dmdt - avg.errs, avg.dmdt + avg.errs,
                color=p[0].get_color(), alpha=.25
            )
        
        if alternative_avg is not None:
            col = style.secondary['all']
            self.ax.fill_between(
                alternative_avg.t,
                alternative_avg.dmdt - alternative_avg.errs,
                alternative_avg.dmdt + alternative_avg.errs,
                color='#333333', alpha=.3
            )

        for series in data:
            col = colors[series.user]
            if full_dmdt is not None:
                s = full_dmdt.filter(user=series.user).first()
                self.ax.plot(
                    s.t, s.dmdt,
                    color=col,
                    linewidth=3,
                    linestyle='--'
                )   

            p = self.ax.plot(
                series.t, series.dmdt,
                color=col,
                linewidth=3,
                label=series.user
            )
            glyphs.append(p[0])

            computed_mark = " "
            if series.computed == True:
                computed_mark += "*"
            elif series.user == "Rietbroek":
                computed_mark += "*"

            labels.append(series.user + computed_mark)
            min_err = "{:.2f}".format(np.nanmin(series.errs))
            max_err = "{:.2f}".format(np.nanmax(series.errs))
            min_errors.append(min_err)
            max_errors.append(max_err)

        # create empty glyph for padding legend:
        empty_glyph = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        empty_label = ''
        legend_header = [[empty_label, 'Contributor', 'Min. Error (Gt/yr)', 'Max. Error (Gt/yr)']]
        legend_rows = [[empty_label, user, _min, _max] for user, _min, _max in zip(labels, min_errors, max_errors)]
        legend_labels = legend_header + legend_rows
        # transpose list
        legend_labels = list(map(list, zip(*legend_labels)))
        # flatten list
        self.labels = list(np.concatenate(legend_labels))

        glyph_header =[[empty_glyph, empty_glyph, empty_glyph, empty_glyph]]
        glyph_rows = glyph_header + [[g] + [empty_glyph]*3 for g in glyphs]
        # transpose list
        glyph_rows = list(map(list, zip(*glyph_rows)))
        self.glyphs = list(np.concatenate(glyph_rows))

        # reduce width of plot by 20% to make space for legend
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width*.6, box.height])

        self.ax.set_ylabel("Rate of Mass Change (Gt/yr)")
        self.ax.set_title(self._sheet_names[region])
        self.fig.autofmt_xdate()

        legend_style = dict(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0., extra=True, ncol=4, handletextpad=-2)
        return "named_dmdt_"+region.value+"_"+group, legend_style

    @render_plot_with_legend
    def named_dm_group_plot(self, region: IceSheet, group: str, data: MassChangeCollection,
                            basis: MassChangeDataSeries=None):
        data = data.filter(user_group=group, basin_id=region)
        self.ax.axhline(0, ls='--', color='k')

        colormap = plt.cm.nipy_spectral
        colorcycle = cycler('color', [colormap(i) for i in np.linspace(0, 1, len(data))])
        self.ax.set_prop_cycle(colorcycle)

        for series in data:
            if basis is not None:
                series = series.align(basis)
            p = self.ax.plot(series.t, series.mass, label=series.user)
            self.glyphs.append(p[0])
            self.labels.append(series.user)

        # reduce width of plot by 20% to make space for legend
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width*.6, box.height])

        self.ax.set_ylabel("Mass Change (Gt)")
        self.ax.set_title(self._sheet_names[region])
        self.fig.autofmt_xdate()

        return "named_dm_" + region.value + "_" + group, dict(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0., extra=True)

    @render_plot_with_legend
    def named_dmdt_comparison_plot(self, data_a: WorkingMassRateCollection, data_b: WorkingMassRateCollection, suffix: str):
        pairs = []
        for series_a in data_a:
            series_b = data_b.filter(
                user=series_a.user, user_group=series_a.user_group,
                basin_id=series_a.basin_id, basin_group=series_a.basin_group
            ).first()

            if series_b is not None:
                pairs.append((series_a, series_b))

        if not pairs:
            return None, {}

        for series_a, series_b in pairs:
            ax = self.ax

            ax.plot(series_a.t, series_a.dmdt, color='r')
            ax.fill_between(
                series_a.t, series_a.dmdt-series_a.errs, series_a.dmdt+series_a.errs, color='r', alpha=.5
            )
            ax.plot(series_b.t, series_b.dmdt, color='g')
            ax.fill_between(
                series_b.t, series_b.dmdt-series_b.errs, series_b.dmdt+series_b.errs, color='g', alpha=.5
            )

            self.glyphs += [self.colour_glyph('r'), self.colour_glyph('g')]
            self.labels += ['from dM', 'from dM/dt']

        # reduce width of plot by 20% to make space for legend
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * .6, box.height])

        self.ax.set_ylabel("Rate of Mass Change (Gt/yr)")
        self.ax.set_title("dM/dt Comparison: %s" % suffix)
        self.fig.autofmt_xdate()

        return "named_dmdt_comparison_" + suffix, dict(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,
                                                                extra=True)

    @render_plot
    def named_dmdt_all(self, regions: Sequence[IceSheet], groups: Sequence[str], data: WorkingMassRateCollection,
                              avg: WorkingMassRateCollection=None, full_dmdt: WorkingMassRateCollection=None,
                              alternative_avg: WorkingMassRateCollection=None, sharex: bool=False, suffix: str=None,
                              flip_grid: bool=False, t_range=None, tag=None) -> str:
        """
        gridded plot of all methods & ice sheets
        """
        if not flip_grid:
            nrows = len(regions)
            ncols = len(groups)
        else:
            nrows = len(groups)
            ncols = len(regions)

        groups_names = {}
        groups_colours = {}

        cyc_markers = cycle("os<v>^<>D")

        names = sorted({s.user for s in data})
        colours = style.UsersColorCollection(names)
        markers = {u: m for u, m in zip(names, cyc_markers)}

        mpl.rc('font', size=18)
        self.fig, axs = plt.subplots(nrows, ncols, sharex=sharex)
        if nrows == 1:
            axs = np.expand_dims(axs, axis=0)
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
        if not flip_grid:
            self.fig.autofmt_xdate()

        self.fig.set_size_inches(ncols*6, nrows*8 + 2)
        plot_n = 0

        for n_group, group in enumerate(groups):
            group_names = sorted({s.user for s in data.filter(user_group=group, basin_id=regions)})

            for n_sheet, sheet in enumerate(regions):
                if not flip_grid:
                    ax_x = n_group
                    ax_y = n_sheet
                else:
                    ax_x = n_sheet
                    ax_y = n_group
                ax = axs[ax_y, ax_x]
                ax.axhline(0, ls='--', color='k')

                if t_range is not None:
                    ax.set_xlim(t_range)

                if tag is None:
                    plot_label = chr(ord('a')+plot_n)
                else:
                    plot_label = tag
                plot_n += 1

                ax.text(
                    .05, .95, plot_label,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes
                )

                if not flip_grid:
                    if ax_y == 0:
                        ax.set_title(self._group_names[group])
                    if ax_y == nrows-1:
                        ax.set_xlabel("Year")

                        patches = [mlines.Line2D([], [], linewidth=3, marker=markers[u], color=colours[u]) for u in group_names]

                        if axs.size == 1:
                            ax.legend(
                                patches, group_names,
                                loc='upper left',
                                bbox_to_anchor=(1, 1),
                                borderaxespad=0,
                                frameon=False
                            )
                        else:
                            ax.legend(
                                patches, group_names,
                                loc='upper center', ncol=2,
                                bbox_to_anchor=(0.5, -0.25),
                                mode="expand", borderaxespad=0.
                            )

                    if ax_x == 0:
                        ax.annotate(self._sheet_names[sheet],
                            xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='center', va='center', rotation=90)
                        y_label = r"Mass Balance (Gt $\mathregular{yr^{-1}}$)"
                        ax.set_ylabel(y_label)
                
                else:
                    if ax_y == 0:
                        ax.set_title(self._sheet_names[sheet])
                    if ax_x == ncols-1:
                        patches = [mlines.Line2D([], [], markersize=10, linewidth=3, marker=markers[u], color=colours[u]) for u in group_names]
                        ax.legend(
                            patches, group_names,
                            loc='upper left',
                            bbox_to_anchor=(1, 1),
                            borderaxespad=0,
                            frameon=False
                        )
                    if ax_y == nrows-1 or flip_grid:
                        ax.set_xlabel("Year")
                    if ax_x == 0:
                        ax.annotate(self._group_names[group],
                            xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='center', va='center', rotation=90)
                        y_label = r"Mass Balance (Gt $\mathregular{yr^{-1}}$)"
                        ax.set_ylabel(y_label)

                if avg is not None:
                    avg_series = avg.filter(user_group=group, basin_id=sheet).first()
                    p = ax.plot(avg_series.t, avg_series.dmdt, label='Average', color='grey', lw=7)

                    bands = ['#dddddd', '#bbbbbb', '#999999']
                    for nsig, c in zip([3, 2, 1], bands):

                        ax.fill_between(
                            avg_series.t,
                            avg_series.dmdt - avg_series.errs * nsig,
                            avg_series.dmdt + avg_series.errs * nsig,
                            color=c  # , alpha=.25
                        )
                
                if alternative_avg is not None:
                    avg_series = alternative_avg.filter(user_group=group, basin_id=sheet).first()
                    col = style.secondary['all']
                    ax.fill_between(
                        avg_series.t,
                        avg_series.dmdt - avg_series.errs,
                        avg_series.dmdt + avg_series.errs,
                        color='#333333', alpha=.3
                    )

                for series in data.filter(user_group=group, basin_id=sheet):
                    col = colours[series.user]
                    m = markers[series.user]

                    if full_dmdt is not None:
                        s = full_dmdt.filter(
                            user=series.user,
                            user_group=group,
                            basin_id=sheet
                        ).first()

                        ax.plot(
                            s.t, s.dmdt,
                            color=col,
                            linewidth=3,
                            linestyle='--'
                        )   

                    ax.plot(
                        series.t, series.dmdt,
                        color=col,
                        linewidth=3,
                        marker=m,
                        label=series.user
                    )
        
        for ax_y, sheet in enumerate(regions):
            reg_min_y = None
            reg_max_y = None
            for ax_x, group in enumerate(groups):
                if flip_grid:
                    ax = axs[ax_x, ax_y]          
                else:
                    ax = axs[ax_y, ax_x]

                ax_min_y, ax_max_y = ax.get_ylim()
                if reg_max_y is None or reg_max_y < ax_max_y:
                    reg_max_y = ax_max_y
                
                if reg_min_y is None or reg_min_y > ax_min_y:
                    reg_min_y = ax_min_y
            
            for ax_x, group in enumerate(groups):
                if flip_grid:
                    ax = axs[ax_x, ax_y]
                else:
                    ax = axs[ax_y, ax_x]

                ax.set_ylim(reg_min_y, reg_max_y)

                if not flip_grid and ax_x > 0:
                    ax.axes.get_yaxis().set_ticklabels([])

        suf = "_share_x" if sharex else ""
        if suffix is not None:
            suf += '_'  + suffix
        return "named_dmdt_all" + suf

    
    @render_plot_with_legend
    def ais_four_panel_plot(self, rate_data: WorkingMassRateCollection, average_rates: WorkingMassRateCollection,
                            sheet_mass: MassChangeCollection):
        """
        four panel plot showing dM/dt per user (coloured by group) for APIS, EAIS, WAIS in first
        three panels, plus aggregated dM(t) per ice sheet coloured for APIS, EAIS, WAIS & AIS in
        final panel
        """

        self.fig, (*sheet_axs, cross_ax) = plt.subplots(4, sharex=True)
        self.fig.autofmt_xdate()
        self.fig.set_size_inches(16, 36)

        ymin = 0
        ymax = 0

        sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais]
        for sheet, ax in zip(sheets, sheet_axs):
            ax.set_title(self._sheet_names[sheet])
            ax.set_ylabel("dM/dt (Gt/yr)")

            for series in rate_data.filter(basin_id=sheet):
                col = style.primary[series.user_group]
                ax.plot(series.t, series.dmdt, color=col)

            series = average_rates.filter(basin_id=sheet).first()
            
            pcol = style.primary['all']
            scol = style.secondary['all']

            ax.plot(series.t, series.dmdt, color=pcol)
            ax.fill_between(
                series.t,
                series.dmdt-series.errs,
                series.dmdt+series.errs,
                color=scol, alpha=.5
            )
            labels = [
                "Altimetry", "Gravimetry", "Mass Budget", "Average"
            ]
            glyphs = [
                self.colour_glyph(style.primary['RA']),
                self.colour_glyph(style.primary['GMB']),
                self.colour_glyph(style.primary['IOM']),
                self.colour_glyph(style.primary['all']),
            ]

            legend = ax.legend(glyphs, labels, loc=3)
            ax.add_artist(legend)

            ax_ymin, ax_ymax = ax.get_ylim()
            ymin = min(ymin, ax_ymin)
            ymax = max(ymax, ax_ymax)

        for ax in sheet_axs:
            ax.set_ylim(ymin, ymax)

        pcols = cycle(["#531A59", "#1B8C6F", "#594508", "#650D1B"])
        scols = cycle(["#9E58A5", "#4CA58F", "#D8B54D", "#A8152E"])

        cross_ax.set_title('Antarctica')
        cross_ax.set_ylabel('Mass Change (Gt)')

        for sheet, pcol, scol in zip([IceSheet.ais]+sheets, pcols, scols):
            series = sheet_mass.filter(basin_id=sheet).first()

            self.labels.append(
                self._sheet_names[sheet]
            )
            self.glyphs.append(
                self.colour_glyph(scol)
            )

            cross_ax.plot(series.t, series.mass, color=pcol)
            cross_ax.fill_between(
                series.t,
                series.mass-series.errs,
                series.mass+series.errs,
                color=scol, alpha=.5
            )
                    
        return "ais_four_panel_plot", dict(loc=3)

    @render_plot
    def annual_dmdt_bars(self, user_rates: WorkingMassRateCollection, sheet_rates: WorkingMassRateCollection, fix_y: bool=False,
                         external_plot: bool=True, imbie1: bool=False, sheets: Sequence[IceSheet]=None, ref_rates: WorkingMassRateCollection=None):
        """
        """
        if sheets is None:
            sheets = [IceSheet.apis, IceSheet.wais, IceSheet.eais]

        interval = 1.
        spacing = .5
        h_width = (interval - spacing) / 2.
        # n_plots = 4 if external_plot else 3
        n_plots = len(sheets)
        counts_above = False

        if external_plot:
            n_plots += 1
            self.fig, (*sheet_axs, final_ax) = plt.subplots(n_plots, sharex=True)
        else:
            self.fig, sheet_axs = plt.subplots(n_plots, sharex=True)
            if n_plots == 1:
                sheet_axs = [sheet_axs]

        self.fig.set_size_inches(16, 9*n_plots)
        mpl.rc('font', size=22)
        ymin = 0
        ymax = 0

        tags = [chr(ord('a') + i) for i, _ in enumerate(sheets)]
        for sheet, ax, label in zip(sheets, sheet_axs, tags):

            if len(tags) > 1:
                ax.text(
                    .05, .95, label,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes
                )

            ax.xaxis.set_tick_params(which='both', top='on', bottom='on', length=8)
            ax.yaxis.set_tick_params(which='both', left='on', right='on', length=8)
            ax.xaxis.set_minor_locator(MultipleLocator(1))

            ax.axhline(0, ls='--', color='k')

            avg_series = sheet_rates.filter(basin_id=sheet).first()

            t_beg = np.floor(avg_series.t.min())
            t_end = np.ceil(avg_series.t.max())

            l_times = []    
            l_texts = []

            print(sheet.value)
            print('year','dmdt','dmdt_sig1')

            for t in np.arange(t_beg, t_end, interval) + .5:
                if not (avg_series.t.min() <= t <= avg_series.t.max()):
                    continue
                l_beg = t - h_width
                l_end = t + h_width

                i = np.argmin(np.abs(avg_series.t - t)) # get index of nearest value
                avg_dmdt = avg_series.dmdt[i]
                avg_sig1 = avg_series.errs[i]
                avg_sig2 = avg_sig1*2
                avg_sig3 = avg_sig1*3

                line = ['%.2f' % n for n in (t, avg_dmdt, avg_sig1)]
                print(','.join(line))

                pcol = style.primary['all']
                scol = style.secondary['all']

                ax.fill_between(
                    [l_beg, l_end],
                    [avg_dmdt-avg_sig3, avg_dmdt-avg_sig3],
                    [avg_dmdt+avg_sig3, avg_dmdt+avg_sig3],
                    color='#dddddd' # , alpha=.25 # alpha=.175
                )
                ax.fill_between(
                    [l_beg, l_end],
                    [avg_dmdt-avg_sig2, avg_dmdt-avg_sig2],
                    [avg_dmdt+avg_sig2, avg_dmdt+avg_sig2],
                    color='#bbbbbb' # scol, alpha=.5 # alpha=.25
                )
                ax.fill_between(
                    [l_beg, l_end],
                    [avg_dmdt-avg_sig1, avg_dmdt-avg_sig1],
                    [avg_dmdt+avg_sig1, avg_dmdt+avg_sig1],
                    color='#999999' # scol, alpha=.75 # alpha=.5
                )
                ax.plot(
                    [l_beg, l_end],
                    [avg_dmdt, avg_dmdt],
                    color='#333333' # pcol
                )
                n_contrib = 0
                n_1_sig = 0
                n_2_sig = 0
                n_3_sig = 0
                outside = 0

                for series in user_rates.filter(basin_id=sheet):
                    t_diff = np.abs(series.t - t)
                    if t_diff.min() > 1.1 * (interval / 2.):
                        continue
                    i = np.argmin(t_diff) # get index of nearest value

                    n_contrib += 1

                    dmdt = series.dmdt[i]

                    pcol = style.primary[series.user_group]

                    ax.plot(
                        [l_beg, l_end],
                        [dmdt, dmdt],
                        color=pcol
                    )
                    if avg_dmdt-avg_sig1 <= dmdt <= avg_dmdt+avg_sig1:
                        n_1_sig += 1
                    if avg_dmdt-avg_sig2 <= dmdt <= avg_dmdt+avg_sig2:
                        n_2_sig += 1
                    if avg_dmdt-avg_sig3 <= dmdt <= avg_dmdt+avg_sig3:
                        n_3_sig += 1
                    else:
                        outside += 1

                if counts_above:
                    l_times.append(t)
                    l_texts.append(str(n_contrib))
                else:
                    bbox_props = dict(
                        boxstyle='circle,pad=0.6',
                        alpha=0, lw=0
                    )

                    text = ax.text(
                        t, avg_dmdt-avg_sig3,
                        str(n_contrib),
                        ha='center', va='top',
                        size=12, bbox=bbox_props
                    )

            if imbie1:
                fname = '~/imbie/imbie1_outputs/imbie_all_{}.x3.csv'.format(sheet.value)
                imbie1_data = pd.read_csv(
                    fname,
                    header=0, index_col=0,
                    usecols=[0, 1, 2],
                    names=['date', 'rate', 'rate_sd']
                )

                ax.plot(imbie1_data.rate, ':k', label='IMBIE 2012')
            
            if ref_rates is not None:
                ref_series = ref_rates.filter(basin_id=sheet).first()

                if ref_series is not None:
                    ax.errorbar(
                        ref_series.t, ref_series.dmdt, yerr=ref_series.errs,
                        lw=0, marker='D', color='k', label='Reference',
                        barsabove=True, capsize=6, elinewidth=2, zorder=1000
                    )

            ax_ymin, ax_ymax = ax.get_ylim()
            ymin = min(ymin, ax_ymin)
            ymax = max(ymax, ax_ymax)

            # ax.set_title(self._sheet_names[sheet])
            p = mpatches.Patch(alpha=0, ec='none')
            l = ax.legend([p], [self._sheet_names[sheet]], loc=1, frameon=False, framealpha=0)
            ax.add_artist(l)

            ax.set_ylabel("dM/dt (Gt $\mathregular{yr^{-1}}$)")
            ax.set_xlabel('Year')
            plt.setp(ax.get_xticklabels(), visible=True)

            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(l_times)
            ax2.set_xticklabels(l_texts, fontsize=12, rotation='vertical')

            if len(tags) == 1:
                box = ax.get_position()
                box = [box.x0, box.y0, box.width, box.height*.9]
                ax.set_position(box)
                ax2.set_position(box)
            # ax2.xaxis.tick_top()

        if fix_y:
            for ax in sheet_axs:
                # ax.set_ylim(ymin, ymax)
                ax.set_ylim(self._imbie1_ylim_dmdt)
        for ax in sheet_axs:
            box = ax.get_position()
            ax2 = ax.twinx()
            ax2.set_position(box)
            _min, _max = ax.get_ylim()
            lims = _min / -360, _max / -360
            ax2.set_ylim(lims)
            ax2.set_ylabel('Sea Level Contribution (mm $\mathregular{yr^{-1}}$)')

        labels = [
            # "Altimetry", "Gravimetry", "Mass Budget", "Average"
            self._group_names['RA'],
            self._group_names['GMB'],
            self._group_names['IOM'],
            self._group_names['all']
        ]
        if imbie1:
            labels.append('IMBIE 2012')
        
        glyphs = [
            mlines.Line2D([], [], color=style.primary['RA']),
            mlines.Line2D([], [], color=style.primary['GMB']),
            mlines.Line2D([], [], color=style.primary['IOM']),
            (mpatches.Patch(color='#999999'),
             mlines.Line2D([], [], color='#333333')),
        ]
        if imbie1:
            glyphs.append(mlines.Line2D([], [], color='k', ls=':'))
        legend_style = dict(frameon=False, framealpha=0)

        if external_plot:
            ax = sheet_axs[0]
            legend_style.update(loc=3, frameon=False, framealpha=0)
            legend = ax.legend(glyphs, labels, **legend_style)
            ax.add_artist(legend)
        else:
            self.fig.legend(
                glyphs, labels,
                loc='upper center', ncol=3 if imbie1 else 2,
                mode='expand', frameon=False
            )

        if external_plot:
            final_ax.xaxis.set_tick_params(which='both', top='on', bottom='on', length=8)
            final_ax.yaxis.set_tick_params(which='both', left='on', right='on', length=8)
            final_ax.xaxis.set_minor_locator(MultipleLocator(1))

            p = mpatches.Patch(alpha=0, ec='none')
            l = final_ax.legend([p], ['Antarctica'], loc=1, frameon=False, framealpha=0)
            final_ax.add_artist(l)
            final_ax.set_ylabel('Mass Change (Gt)')

            final_ax.text(
                .05, .95, 'd',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )

            supl_data = pd.read_csv(
                '~/Downloads/ais_data_out.txt',
                header=None,
                names=["year_ais",
                    "dm_eais",
                    "dm_eais_sd",
                    "dm_wais",
                    "dm_wais_sd",
                    "dm_apis",
                    "dm_apis_sd",
                    "dm_ais",
                    "dm_ais_sd"],
                delim_whitespace=True
            )
            names = ['ais', 'wais', 'eais', 'apis']
            pcols = cycle(["#531A59", "#1B8C6F", "#594508", "#650D1B"])
            scols = cycle(["#9E58A5", "#4CA58F", "#D8B54D", "#A8152E"])
            t = supl_data.year_ais

            full_names = {
                'ais': 'Antarctica',
                'wais': 'West Antarctica',
                'eais': 'East Antarctica',
                'apis': 'Antarctic Peninsula'
            }
            labels = []
            glyphs = []

            for name, pcol, scol in zip(names, pcols, scols):
                dm = getattr(supl_data, 'dm_'+name)
                sd = getattr(supl_data, 'dm_'+name+'_sd')

                final_ax.plot(t, dm, color=pcol)
                final_ax.fill_between(
                    t,
                    dm-sd,
                    dm+sd,
                    color=scol,
                    alpha=.5
                )
                glyphs.append(
                    self.colour_glyph(pcol)
                )
                labels.append(
                    full_names[name]
                )


            l = final_ax.legend(glyphs, labels, loc=3, frameon=False, framealpha=0)
            final_ax.add_artist(l)

            ax2 = final_ax.twinx()
            _min, _max = final_ax.get_ylim()
            lims = _min / -360, _max / -360
            ax2.set_ylim(lims)
            ax2.set_ylabel('Sea Level Contribution (mm)')


        suffix = "_fixed_y" if fix_y else ""
        if external_plot:
            suffix += "_ext"
        suffix += '_'+'_'.join(sheet.value for sheet in sheets)
        return "annual_dmdt_bars"+suffix

    @render_plot
    def greenland_plot(self, data: MassChangeDataSeries, imbie1: bool=False) -> str:
        from matplotlib.ticker import MultipleLocator

        self.ax.xaxis.set_tick_params(which='both', top='on', bottom='on', length=8)
        self.ax.yaxis.set_tick_params(which='both', left='on', right='on', length=8)
        self.ax.xaxis.set_minor_locator(MultipleLocator(1))

        self.ax.set_title('Greenland')
        self.ax.set_ylabel('Mass Change (Gt)')

        pcol = "#531A59"
        scol = "#9E58A5"

        p, = self.ax.plot(data.t, data.mass, color=pcol)
        self.ax.fill_between(
            data.t,
            data.mass-data.errs,
            data.mass+data.errs,
            color=scol,
            alpha=.5
        )
        glyphs = [
            (self.colour_glyph(scol, alpha=.5), p)
        ]
        labels = ["IMBIE 2018"]

        if imbie1:

            fname = '~/imbie/imbie1_outputs/imbie_all_gris.x3.csv'
            imbie1_data = pd.read_csv(
                fname,
                header=0, index_col=0,
                usecols=[0, 3, 4],
                names=['date', 'mass', 'mass_sd']
            )

            self.ax.plot(
                imbie1_data.date.values,
                imbie1_data.mass.values,
                ls=':', color=pcol,
                label='IMBIE 2012'
            )

        
            glyphs.append(
                mlines.Line2D([], [], ls=':', color='k')
            )
            labels.append(
                'IMBIE 2012'
            )

        l = self.ax.legend(glyphs, labels, loc=3, frameon=False, framealpha=0)
        self.ax.add_artist(l)
        self.ax.set_ylim(self._imbie1_ylim_dm)

        ax2 = self.ax.twinx()
        _min, _max = self.ax.get_ylim()
        lims = _min / -360, _max / -360
        ax2.set_ylim(lims)
        ax2.set_ylabel('Sea Level Contribution (mm)')

        return "gris_mass_comparison"

    @render_plot
    def external_data_plot(self, imbie1: bool=False) -> str:
        from matplotlib.ticker import MultipleLocator

        self.ax.xaxis.set_tick_params(which='both', top='on', bottom='on', length=8)
        self.ax.yaxis.set_tick_params(which='both', left='on', right='on', length=8)
        self.ax.xaxis.set_minor_locator(MultipleLocator(1))

        self.ax.set_title('Antarctica')
        self.ax.set_ylabel('Mass Change (Gt)')

        supl_data = pd.read_csv(
            '~/Downloads/ais_data_out.txt',
            header=None,
            names=["year_ais",
                "dm_eais",
                "dm_eais_sd",
                "dm_wais",
                "dm_wais_sd",
                "dm_apis",
                "dm_apis_sd",
                "dm_ais",
                "dm_ais_sd"],
            delim_whitespace=True
        )
        names = ['ais', 'wais', 'eais', 'apis']
        pcols = cycle(["#531A59", "#1B8C6F", "#594508", "#650D1B"])
        scols = cycle(["#9E58A5", "#4CA58F", "#D8B54D", "#A8152E"])
        t = supl_data.year_ais

        full_names = {
            'ais': 'Antarctica',
            'wais': 'West Antarctica',
            'eais': 'East Antarctica',
            'apis': 'Antarctic Peninsula'
        }
        labels = []
        glyphs = []

        for name, pcol, scol in zip(names, pcols, scols):
            dm = getattr(supl_data, 'dm_'+name)
            sd = getattr(supl_data, 'dm_'+name+'_sd')

            p, = self.ax.plot(t, dm, color=pcol)
            self.ax.fill_between(
                t,
                dm-sd,
                dm+sd,
                color=scol,
                alpha=.5
            )
            glyphs.append(
                (self.colour_glyph(scol, alpha=.5), p)
            )
            labels.append(
                full_names[name]
            )

            if name == 'ais' or not imbie1:
                continue

            fname = '~/imbie/imbie1_outputs/imbie_all_{}.x3.csv'.format(name)
            imbie1_data = pd.read_csv(
                fname,
                header=0, index_col=0,
                usecols=[0, 3, 4],
                names=['date', 'mass', 'mass_sd']
            )

            self.ax.plot(imbie1_data.mass, ls=':', color=pcol, label='IMBIE 2012')

        if imbie1:
            glyphs.append(
                mlines.Line2D([], [], ls=':', color='k')
            )
            labels.append(
                'IMBIE 2012'
            )

        l = self.ax.legend(glyphs, labels, loc=3, frameon=False, framealpha=0)
        self.ax.add_artist(l)
        self.ax.set_ylim(self._imbie1_ylim_dm)

        ax2 = self.ax.twinx()
        _min, _max = self.ax.get_ylim()
        lims = _min / -360, _max / -360
        ax2.set_ylim(lims)
        ax2.set_ylabel('Sea Level Contribution (mm)')

        return "external_data_plot"

    @render_plot
    def discharge_plot(self, mean_discharge: MassChangeDataSeries, groups_discharge: MassChangeCollection,
                       users_discharge: MassChangeCollection) -> str:
        self.ax.fill_between(
            mean_discharge.t,
            mean_discharge.mass-mean_discharge.errs,
            mean_discharge.mass+mean_discharge.errs,
            facecolor=style.colours.secondary['all'],
            edgecolor=style.colours.primary['all'],
            alpha=.5, lw=1
        )
        self.ax.plot(
            mean_discharge.t, mean_discharge.mass,
            lw=2, color=style.colours.primary['all'],
            label='All'
        )

        for group in 'IOM', 'RA', 'GMB':
            discharge = groups_discharge.filter(
                user_group=group
            ).first()

            if discharge is None:
                continue

            self.ax.fill_between(
                discharge.t,
                discharge.mass-discharge.errs,
                discharge.mass+discharge.errs,
                facecolor=style.colours.secondary[group],
                edgecolor=style.colours.primary[group],
                alpha=.5, lw=1
            )
            self.ax.plot(
                discharge.t, discharge.mass,
                lw=2, color=style.colours.primary[group],
                label=self._group_names[group]
            )

        for i, series in enumerate(users_discharge):
            self.ax.fill_between(
                series.t,
                series.mass-series.errs,
                series.mass+series.errs,
                edgecolor='#ee33aa', lw=1,
                facecolor='#ee33aa', alpha=.5
            )

            self.ax.plot(
                series.t, series.mass,
                lw=2, color='#ee33aa',
                label=series.user
            )

        l = self.ax.legend(loc=3, frameon=False, framealpha=0)
        self.ax.add_artist(l)
        self.ax.set_ylabel('Discharge (Gt)')
        self.ax.set_title('Greenland')

        ax2 = self.ax.twinx()
        _min, _max = self.ax.get_ylim()
        lims = _min / -360, _max / -360
        ax2.set_ylim(lims)
        ax2.set_ylabel('Sea Level Contribution (mm)')

        return "discharge_plot"

    @render_plot
    def discharge_comparison_plot(self, mass_balance: MassChangeDataSeries, surface_mass: MassChangeDataSeries, discharge: MassChangeDataSeries) -> str:
        self.ax.fill_between(
            mass_balance.t,
            mass_balance.mass-mass_balance.errs,
            mass_balance.mass+mass_balance.errs,
            alpha=.5, color='#9E58A5'
        )
        self.ax.plot(
            mass_balance.t, mass_balance.mass, color='#531A59', label='IMBIE 2018'
        )
        self.ax.fill_between(
            surface_mass.t,
            surface_mass.mass-surface_mass.errs,
            surface_mass.mass+surface_mass.errs,
            alpha=.5, color='#4CA58F'
        )
        self.ax.plot(
            surface_mass.t, surface_mass.mass, color='#1B8C6F', label='SMB'
        )
        self.ax.fill_between(
            discharge.t,
            discharge.mass-discharge.errs,
            discharge.mass+discharge.errs,
            alpha=.5, color='#594508'
        )
        self.ax.plot(
            discharge.t, discharge.mass, color='#D8B54D', label='Dynamics'
        )
        self.ax.legend(loc='best', frameon=False, framealpha=0)
        self.ax.set_ylabel('Mass Change (Gt)')
        self.ax.set_xlabel('Year')

        self.fig.autofmt_xdate()
        return 'imbie_smb_dynamics'

    @render_plot
    def discharge_scatter_plot(self, reference: MassChangeDataSeries, data: MassChangeCollection) -> str:
        for series in data:
            ref_mass = np.interp(
                series.t, reference.t, reference.mass
            )
            lab = self._group_names[series.user_group]
            col = style.colours.primary[series.user_group]

            self.ax.scatter(
                series.mass, ref_mass,
                marker='o',
                color=col,
                label=lab
            )
        
        margin = .05

        min_x, max_x = self.ax.get_xlim()
        min_y, max_y = self.ax.get_ylim()

        min_ax = min(min_x, min_y)
        max_ax = max(max_x, max_y)
        diff = max_ax - min_ax

        min_ax += margin * diff
        max_ax += margin * diff

        self.ax.plot(
            [min_ax, max_ax],
            [min_ax, max_ax],
            ls='--', c='.3'
        )
        self.ax.set_xlim((min_ax, max_ax))
        self.ax.set_ylim((min_ax, max_ax))
        
        self.ax.legend(loc='best', frameon=False, framealpha=0)
        self.ax.set_ylabel('Reference Discharge (Gt)')
        self.ax.set_xlabel('Discharge (Gt)')
        self.ax.set_title('Greenland')

        return "discharge_scatter_plot"

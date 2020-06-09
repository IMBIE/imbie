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
    def named_dmdt_group_plot(self, region: IceSheet, group: str, data: WorkingMassRateCollection, avg: WorkingMassRateDataSeries=None):
        data = data.filter(user_group=group, basin_id=region)

        colormap = plt.cm.nipy_spectral
        colorcycle = cycler('color', [colormap(i) for i in np.linspace(0, 1, len(data))])
        self.ax.set_prop_cycle(colorcycle)

        if avg is not None:
            p = self.ax.plot(avg.t, avg.dmdt, ls='--', label='Average')
            self.glyphs.append(p[0])
            self.labels.append('Average')

            self.ax.fill_between(
                avg.t, avg.dmdt - avg.errs, avg.dmdt + avg.errs,
                color=p[0].get_color(), alpha=.5
            )

        for series in data:
            p = self.ax.plot(series.t, series.dmdt, label=series.user)
            self.glyphs.append(p[0])
            self.labels.append(series.user)

        # reduce width of plot by 20% to make space for legend
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width*.6, box.height])

        self.ax.set_ylabel("Rate of Mass Change (Gt/yr)")
        self.ax.set_title(self._sheet_names[region])
        self.fig.autofmt_xdate()

        return "named_dmdt_"+region.value+"_"+group, dict(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0., extra=True)

    @render_plot_with_legend
    def named_dm_group_plot(self, region: IceSheet, group: str, data: MassChangeCollection,
                            basis: MassChangeDataSeries=None):
        data = data.filter(user_group=group, basin_id=region)

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
            return "", {}


        # colormap = plt.cm.nipy_spectral
        # colorcycle = (colormap(i) for i in np.linspace(0, 1, len(pairs)))
        #
        # size = int(np.sqrt(len(pairs)) + 1)
        # fig, axes = plt.subplots(size, size, True, True)

        # for color, (series_a, series_b), ax in zip(colorcycle, pairs, axes.flat):
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

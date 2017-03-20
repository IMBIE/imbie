import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
import matplotlib as mpl
import numpy as np
import math
import os
from typing import Sequence, Tuple

from . import plots
from . import style

from functools import wraps
from itertools import cycle, chain
from collections import OrderedDict

from imbie2.const.basins import *
from imbie2.const import AverageMethod
from imbie2.model.collections import MassChangeCollection, MassRateCollection, WorkingMassRateCollection
from imbie2.util.functions import ts2m, move_av, match, t2m
from imbie2.util.combine import weighted_combine as ts_combine
from imbie2.model.managers import MassChangeCollectionsManager, MassRateCollectionsManager
from imbie2.model.series import *


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
    def wrapped(obj, *args, **kwargs):
        obj.clear_plot()
        obj.clear_legend()
        ret, leg = method(obj, *args, **kwargs)
        obj.draw_legend(**leg)
        obj.draw_plot(ret)

    return wrapped


class Plotter:
    _time0 = 1990
    _time1 = 2020
    _dmdt0 = -700
    _dmdt1 = 300
    _dm0 = -7000
    _dm1 = 2000
    _set_limits = False

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
        "IOM": "Mass Budget",
        "all": "All"
    }

    def __init__(self, filetype=None, path=None, limits: bool=None):
        self._ext = filetype
        if path is None:
            path = os.getcwd()
        self._path = os.path.expanduser(path)

        if limits is not None:
            self._set_limits = limits

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

    def draw_plot(self, fname=None):
        if fname is None:
            return # self.clear_plot()
        elif self._ext is None:
            plt.show()
        else:
            if not os.path.exists(self._path):
                print("creating directory: {}".format(self._path))
                os.makedirs(self._path)
            fname = fname+'.'+self._ext
            fpath = os.path.join(self._path, fname)
            plt.savefig(fpath, dpi=192)
            self.ax.clear()
            self.fig.clear()
            #
            print(fpath)

    def clear_plot(self):
        self.ax = plt.gca()
        self.fig = plt.gcf()

    def draw_legend(self, **legend_opts):
        if (not self.glyphs) or (not self.labels):
            return
        if 'parent' in legend_opts:
            if legend_opts.pop('parent') == 'fig':
                self.fig.legend(self.glyphs, self.labels, **legend_opts)
                return
        plt.legend(self.glyphs, self.labels, **legend_opts)

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
    def colour_glyph(colour, label=None):
        return mpatches.Patch(color=colour, label=label)

    @render_plot_with_legend
    def sheets_time_bars(self, data, sheets, names, *groups):
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

            min_t = {}
            max_t = {}
            group = {}
            bsets = {}

            for series in data[sheet]:
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
        plt.setp(yticklabels, visible=False)

        leg_params = {'loc': 'lower center', 'ncol': 3, 'parent': 'fig'}
                      # {'loc': 4, 'frameon': False}
        self.fig.set_size_inches(16, 9)
        sheet_names = "_".join(s.value for s in sheets)
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

    @render_plot_with_legend
    def rignot_vs_zwally_scatter(self, data, *sheets):
        _min = None
        _max = None

        rms_all = 0
        rms_sub = 0
        num_all = 0
        num_sub = 0

        if not sheets:
            suffix = "all"
            sheets = iter(IceSheet)
        else:
            suffix = "_".join(s.value for s in sheets)

        for sheet in sheets:
            z_rate = {}
            r_rate = {}
            z_errs = {}
            r_errs = {}
            groups = {}

            rate = [r_rate, z_rate]
            errs = [r_errs, z_errs]
            pair = [BasinGroup.rignot,
                    BasinGroup.zwally]

            self.labels = []
            self.glyphs = []

            for cat, r, e in zip(pair, rate, errs):
                for series in data[sheet]:
                    if series.basin_group != cat:
                        continue
                    u = series.user
                    groups[u] = series.user_group

                    r[u] = series.mean
                    e[u] = series.sigma

            z_vals = []
            r_vals = []

            for u in z_rate:
                if u not in r_rate: continue

                z_vals.append(z_rate[u])
                r_vals.append(r_rate[u])

                sqr_diff = (z_rate[u] - r_rate[u]) ** 2

                rms_all += sqr_diff
                num_all += 1
                if u != "Blazquez":
                    rms_sub += sqr_diff
                    num_sub += 1

                z_mean = [z_rate[u], z_rate[u]]
                z_line = [z_rate[u] - z_errs[u],
                          z_rate[u] + z_errs[u]]
                r_mean = [r_rate[u], r_rate[u]]
                r_line = [r_rate[u] - r_errs[u],
                          r_rate[u] + r_errs[u]]
                g = groups[u]
                grp = self._group_names[g]
                if grp not in self.labels:
                    self.labels.append(grp)
                    self.glyphs.append(
                        self.group_glyph(g)
                    )

                c = style.colours.primary[g]

                self.ax.plot(z_mean, r_line, color=c)
                self.ax.plot(z_line, r_mean, color=c)
                if suffix != "all":
                    self.ax.text(z_rate[u], r_rate[u], ' '+u)

                if _min is None or min(r_line) < _min:
                    _min = min(r_line)
                if _max is None or max(r_line) > _max:
                    _max = max(r_line)
                if _min is None or min(z_line) < _min:
                    _min = min(z_line)
                if _max is None or max(z_line) > _max:
                    _max = max(z_line)
            self.ax.scatter(z_vals, r_vals)

        self.ax.plot([_min, _max], [_min, _max], color='k')
        self.ax.set_xlabel('Zwally dM/dt (Gt/yr)')
        self.ax.set_ylabel('Rignot dM/dt (Gt/yr)')
        self.ax.set_xlim(_min, _max)
        self.ax.set_ylim(_min, _max)

        self.ax.grid()

        # print(suffix, math.sqrt(rms_all / num_all), math.sqrt(rms_sub / num_sub))

        self.fig.set_size_inches(16, 9)
        return "rignot_vs_zwally_scatter_"+suffix, {'loc': 4, 'frameon': False}

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
                          methods: Sequence[str], sheets: Sequence[IceSheet]):
        # get mean & error dM/dt per ice-sheet and group
        width = len(methods)
        min_y = None
        max_y = None
        max_x = 0

        for i, sheet in enumerate(sheets):
            max_x += width+1
            # plot all-group patches
            sheet_series = sheet_avgs.filter(basin_id=sheet).first()
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
                ).first()

                mean = group_series.mean
                err = group_series.sigma

                jx = x + j + .5
                y1 = mean - err
                y2 = mean + err

                self.ax.plot(
                    [jx, jx], [y1, y2],
                    linewidth=10,
                    color=style.colours.primary[method]
                    # marker='_'
                )
                if y1 < min_y: min_y = y1
                if y2 > max_y: max_y = y2
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
        # turn on minor ticks (both axes), then disable x-axis minor ticks
        # self.ax.minorticks_on()
        # self.ax.tick_params(axis='x', which='minor', top='off')
        self.ax.tick_params(
            axis='y', which='both', left='off', labelleft='off', right='off')

        self.ax.set_xlim(-.5, max_x-.5)
        self.ax.set_ylim(min_y - y_margin, max_y + y_margin)

        # plt.ylabel("Mass Balance (Gt/yr)", fontweight='bold')
        plt.xlabel("Ice Sheet", fontweight='bold')
        self.fig.set_size_inches(16, 9)

        return "sheets_error_bars", {'frameon': False}

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
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))
        if mark is None:
            mark = []

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)

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
                        x = contrib.t[-1]
                        y = contrib.dmdt[-1]
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
                                   regions, suffix: str=None, mark: Sequence[str]=None) -> str:
        if mark is None:
            mark = []
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)

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
                                    regions) -> str:
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)

            x_avg = region_avgs.filter(basin_id=name).average()
            if x_avg is None:
                print(name)
                continue

            pcol = style.colours.primary["all"]
            scol = style.colours.secondary["all"]

            self.ax.plot(x_avg.t, x_avg.dmdt, color=pcol)
            self.ax.fill_between(
                x_avg.t, x_avg.dmdt - x_avg.errs, x_avg.dmdt + x_avg.errs,
                color=scol, alpha=.5
            )

            for g_avg in group_avgs.filter(basin_id=name):
                pcol = style.colours.primary[g_avg.user_group]
                scol = style.colours.secondary[g_avg.user_group]

                self.ax.plot(g_avg.t, g_avg.dmdt, color=pcol)
                self.ax.fill_between(
                    g_avg.t, g_avg.dmdt - g_avg.errs, g_avg.dmdt + g_avg.errs,
                    color=scol, alpha=.5
                )

            # get start & end time of common period
            com_t_min = group_avgs.concurrent_start()
            com_t_max = group_avgs.concurrent_stop()
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

        self.fig.suptitle("dM/dt intercomparison")
        self.fig.autofmt_xdate()

        name = "groups_rate_intercomparison_"
        name += "_".join([s.value for s in regions])
        return name

    @render_plot
    def groups_mass_intercomparison(self, region_avgs: MassChangeCollection, group_avgs: MassChangeCollection,
                                    regions) -> str:
        plt_w, plt_h, plt_shape = self._get_subplot_shape(len(regions))

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(plt_shape+i+1)

            x_avg = region_avgs.filter(basin_id=name).average()
            if x_avg is None:
                print(name)
                continue

            pcol = style.colours.primary["all"]
            scol = style.colours.secondary["all"]

            self.ax.plot(x_avg.t, x_avg.mass, color=pcol)
            self.ax.fill_between(
                x_avg.t, x_avg.mass - x_avg.errs, x_avg.mass + x_avg.errs,
                color=scol, alpha=.5
            )

            for g_avg in group_avgs.filter(basin_id=name):
                pcol = style.colours.primary[g_avg.user_group]
                scol = style.colours.secondary[g_avg.user_group]

                self.ax.plot(g_avg.t, g_avg.mass, color=pcol)
                self.ax.fill_between(
                    g_avg.t, g_avg.mass - g_avg.errs, g_avg.mass + g_avg.errs,
                    color=scol, alpha=.5
                )

            # get start & end time of common period
            com_t_min = group_avgs.concurrent_start()
            com_t_max = group_avgs.concurrent_stop()
            # plot v. lines to show period
            self.ax.axvline(com_t_min, ls='--', color='k')
            self.ax.axvline(com_t_max, ls='--', color='k')

            # set title & axis labels
            # self.ax.set_ylabel("Mass Change (Gt)")
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
        pcols = cycle(["#531A59", "#1B8C6F", "#594508"])
        scols = cycle(["#9E58A5", "#4CA58F", "#D8B54D"])

        for region, pcol, scol in zip(regions, pcols, scols):
            avg = region_avgs.filter(basin_id=region).average()

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
        self.ax.set_xticklabels(self.ax.xaxis.get_majorticklabels(), rotation=45)

        return "regions_mass_intercomparison_"+"_".join(r.value for r in regions), {"frameon": False, "loc": 3}
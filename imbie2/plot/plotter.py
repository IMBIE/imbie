import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
import matplotlib as mpl
import numpy as np
import math
import os
from typing import Sequence

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

    def __init__(self, filetype=None, path=None):
        self._ext = filetype
        if path is None:
            path = os.getcwd()
        self._path = os.path.expanduser(path)

    def draw_plot(self, fname=None):
        if fname is None:
            return # self.clear_plot()
        elif self._ext is None:
            plt.show()
        else:
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
    def sheets_error_bars(self, data, t_min=None, t_max=None):
        # get mean & error dM/dt per ice-sheet and group
        max_methods = 0
        sorted_mean = OrderedDict()
        sorted_errs = OrderedDict()
        all_methods = []

        for sheet in IceSheet:
            if sheet == IceSheet.ais: continue
            methods = {}

            _any = False
            for series in data[sheet]:
                # if (t_min is not None and series.min_time > t_min) or\
                #    (t_max is not None and series.max_time < t_max):
                #     continue
                _any = True
                meth = series.user_group
                if meth not in all_methods:
                    all_methods.append(meth)

                if meth not in methods:
                    methods[meth] = []
                methods[meth].append(series)

            if not _any: continue

            if len(methods) > max_methods:
                max_methods = len(methods)

            all_means = []
            all_sigma = []

            sorted_errs[sheet] = OrderedDict()
            sorted_mean[sheet] = OrderedDict()

            for meth in methods:
                means = [s.mean for s in methods[meth]]
                sigma = [s.sigma for s in methods[meth]]

                mean = np.mean(means)
                errs = math.sqrt(
                    np.mean(np.square(sigma))
                )

                sorted_mean[sheet][meth] = mean
                sorted_errs[sheet][meth] = errs

                all_means.append(mean)
                all_sigma.append(errs)
            # compute mean & error per ice-sheet across all groups
            sorted_mean[sheet]['all'] = np.mean(
                all_means
            )
            sorted_errs[sheet]['all'] = math.sqrt(
                np.mean(np.square(all_sigma))
            )
        sorted_mean[IceSheet.ais] = OrderedDict()
        sorted_errs[IceSheet.ais] = OrderedDict()
        for g in 'IOM', 'GMB', 'RA', 'all':
            others = IceSheet.apis, IceSheet.wais, IceSheet.eais
            sorted_mean[IceSheet.ais][g] = sum(sorted_mean[o][g] for o in others)
            sorted_errs[IceSheet.ais][g] = sum(sorted_errs[o][g] for o in others)

        width = len(all_methods)
        min_y = None
        max_y = None
        max_x = 0

        for i, sheet in enumerate(IceSheet):
            if sheet not in sorted_mean: continue

            max_x += width+1
            # plot all-group patches
            mean = sorted_mean[sheet]['all']
            err = sorted_errs[sheet]['all']

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
            for j, method in enumerate(all_methods):
                if method not in sorted_mean[sheet]:
                    continue

                mean = sorted_mean[sheet][method]
                err = sorted_errs[sheet][method]

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
        for group in all_methods:
            self.glyphs.append(
                self.group_glyph(group)
            )
            self.labels.append(
                self._group_names[group]
            )

        # set axis labels, limits, ticks
        names = [self._sheet_names[s] for s in IceSheet]
        ticks = [(i+.5)*width+i for i, _ in enumerate(IceSheet)]

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
    def box_coverage(self, data, *groups):
        if not groups:
            groups = "RA", "GMB", "IOM"

        for i, sheet in enumerate(IceSheet):
            ax = plt.subplot(230 + i + 1)

            sheet_data = data[sheet]
            if not sheet_data:
                return None, {}
            plots.coverage_boxes(ax, sheet_data, groups)

            ax.set_title(self._sheet_names[sheet] + "\n")
            ax.set_ylabel("Mass Balance (Gt/yr)")
            ax.set_ylim(self._dmdt0, self._dmdt1)

        return "box_coverage_"+'_'.join(groups)

    @render_plot_with_legend
    def group_boxes(self, data, group):

        pcol = style.colours.primary[group]
        scol = style.colours.secondary[group]

        for sheet in IceSheet:
            for series in data[sheet]:
                if series.user_group != group:
                    continue

                mean = series.mean
                sigma = series.sigma

        # TODO: finish this
        raise NotImplemented()

    @render_plot_with_legend
    def rate_series_sheet_groups(self, data, sheet, groups=None):
        sheet_name = sheet.value
        title = self._sheet_names[sheet]
        if groups is None:
            groups = 'RA', 'GMB', 'IOM'
        groups_rate = {g: [] for g in groups}
        groups_errs = {g: [] for g in groups}
        groups_time = {g: [] for g in groups}
        groups_rate["all"] = []
        groups_errs["all"] = []
        groups_time["all"] = []
        t_min = {g: None for g in groups}
        t_max = {g: None for g in groups}

        if sheet == IceSheet.ais:
            sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais]
        else: sheets = [sheet]

        for sheet in sheets:
            sheet_rate = {g: [] for g in groups}
            sheet_errs = {g: [] for g in groups}
            sheet_time = {g: [] for g in groups}

            for series in data[sheet]:
                g = series.user_group
                if g not in groups:
                    continue

                t, dmdt, errs = chunk_rates(series)

                sheet_time[g].append(t)
                sheet_rate[g].append(dmdt)
                sheet_errs[g].append(errs)

                if t_min[g] is None or t_min[g] > series.min_time:
                    t_min[g] = series.min_time
                if t_max[g] is None or t_max[g] < series.max_time:
                    t_max[g] = series.max_time

            all_time = []
            all_rate = []
            all_errs = []
            for g in groups:
                t, rate = ts_combine(sheet_time[g], sheet_rate[g])
                _, errs = ts_combine(sheet_time[g], sheet_errs[g], error=True)

                groups_time[g].append(t)
                groups_rate[g].append(rate)
                groups_errs[g].append(errs)
                all_time.append(t)
                all_rate.append(rate)
                all_errs.append(errs)

            t, rate = ts_combine(all_time, all_rate)
            _, errs = ts_combine(all_time, all_errs, error=True)

            groups_time["all"].append(t)
            groups_rate["all"].append(rate)
            groups_errs["all"].append(errs)

        for g in chain(groups, ["all"]):

            if len(sheets) > 1:
                t, rate = sum_sheets(groups_time[g], groups_rate[g])
                _, errs = sum_sheets(groups_time[g], groups_errs[g])
            else:
                t = groups_time[g][0]
                rate = groups_rate[g][0]
                errs = groups_errs[g][0]

            col_a = style.colours.primary[g]
            col_b = style.colours.secondary[g]

            self.ax.plot(t, rate, color=col_a)
            self.ax.fill_between(
                t, rate-errs, rate+errs,
                color=col_b, alpha=0.5,
                interpolate=True
            )
            # add legend values
            self.glyphs.append(
                self.group_glyph(g)
            )
            self.labels.append(
                self._group_names[g]
            )

        com_t_min = max(t_min.values())
        com_t_max = min(t_max.values())
        self.ax.axvline(com_t_min, ls='--', color='k')
        self.ax.axvline(com_t_max, ls='--', color='k')

        plt.title(title)
        plt.ylabel("Mass Balance (Gt/yr)")
        self.ax.set_ylim(self._dmdt0, self._dmdt1)
        self.ax.set_xlim(self._time0, self._time1)
        return "rate_series_sheet_groups_"+sheet_name, {'frameon': False}

    @render_plot_with_legend
    def mass_series_continent_sheets(self, data, sheet_groups, names, min_t=None,
                                     max_t=None, avg_method=AverageMethod.equal_groups):

        pcols = cycle(["#531A59", "#1B8C6F", "#594508"])
        scols = cycle(["#9E58A5", "#4CA58F", "#D8B54D"])

        for name, sheet_set, pcol, scol in zip(names, sheet_groups, pcols, scols):
            set_time = []
            set_dmdt = []
            set_errs = []

            for sheet in sheet_set:
                groups_time = {}
                groups_dmdt = {}
                groups_errs = {}
                sheet_time = []
                sheet_dmdt = []
                sheet_errs = []

                for series in data[sheet]:
                    if avg_method == AverageMethod.equal_groups:
                        g = series.user_group
                    else:
                        g = "all"
                    if g not in groups_time:
                        groups_time[g] = []
                        groups_dmdt[g] = []
                        groups_errs[g] = []

                    t, dmdt, errs = chunk_rates(series)
                    groups_time[g].append(t)
                    groups_dmdt[g].append(dmdt)
                    groups_errs[g].append(errs)

                for g in groups_time:
                    if avg_method == AverageMethod.inverse_errs:
                        w = [1./errs for errs in groups_errs[g]]
                        t, dmdt = ts_combine(groups_time[g], groups_dmdt[g], w)
                        _, errs = ts_combine(groups_time[g], groups_errs[g], error=True)
                    else:
                        t, dmdt = ts_combine(groups_time[g], groups_dmdt[g])
                        _, errs = ts_combine(groups_time[g], groups_errs[g], error=True)

                    sheet_time.append(t)
                    sheet_dmdt.append(dmdt)
                    sheet_errs.append(errs)

                if len(sheet_time) > 1:
                    t, dmdt = ts_combine(sheet_time, sheet_dmdt)
                    _, errs = ts_combine(sheet_time, sheet_errs, error=True)
                else:
                    t = sheet_time[0]
                    dmdt = sheet_dmdt[0]
                    errs = sheet_errs[0]

                set_time.append(t)
                set_dmdt.append(dmdt)
                set_errs.append(errs)

            # t, _ = ts_combine(set_time, set_dmdt)
            #
            # dmdt = np.zeros(t.shape, dtype=np.float64)
            # errs = np.zeros(t.shape, dtype=np.float64)
            #
            # for i, times in enumerate(set_time):
            #     i1, i2 = match(t, times)
            #     dmdt[i1] += set_dmdt[i][i2]
            #     errs[i1] += set_errs[i][i2]

            t, dmdt = sum_sheets(set_time, set_dmdt)
            _, errs = sum_sheets(set_time, set_errs)

            ok = np.ones(t.shape, dtype=bool)
            if min_t is not None:
                ok = np.logical_and(
                    ok, t > min_t
                )
            if max_t is not None:
                ok = np.logical_and(
                    ok, t < max_t
                )

            t = t[ok]
            mass = np.cumsum(dmdt[ok]) / 12.
            errs = np.cumsum(errs[ok]) / 12.

            self.ax.plot(t, mass, color=pcol)
            self.ax.fill_between(
                t, mass-errs, mass+errs,
                alpha=0.5, color=scol,
                interpolate=True
            )
            # add legend values
            self.glyphs.append(
                self.colour_glyph(pcol)
            )
            self.labels.append(name)

        plt.ylabel("Mass Change (Gt)")
        avg_suffixes = {
            AverageMethod.equal_groups: "eqg_",
            AverageMethod.equal_series: "eqs_",
            AverageMethod.inverse_errs: "inv_",
            AverageMethod.split_altimetry: "qrt_"
        }
        suffix = avg_suffixes[avg_method]

        lims = ""
        t0, t1 = self._time0, self._time1
        if min_t is not None:
            lims += str(min_t) + '_'
            t0 = min_t
        if max_t is not None:
            lims += str(max_t) + '_'
            t1 = max_t

        sheets = "_".join(names)
        self.ax.set_ylim(self._dm0, self._dm1)
        self.ax.set_xlim(t0, t1)
        return "continent_"+suffix+lims+sheets, {"frameon": False, "loc": 3}

    @render_plot_with_legend
    def sheet_scatter(self, ice_sheet, basins, data):
        markers = cycle(style.markers)
        groups = set()
        legend_style = {
            'frameon': False,
            'prop': {'size': 8}
        }

        for marker, basin in zip(markers, basins.sheet(ice_sheet)):
            if basin not in data: continue

            basin_data = data[basin]
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
    def sheet_methods_contributing_integrated(self, data, *groups):
        if not groups:
            groups = None

        for i, sheet in enumerate(IceSheet):
            self.ax = plt.subplot(230 + i + 1)
            self._sheet_methods_contributing_integrated(data, sheet, groups)

        if groups is None:
            groups = "RA", "GMB", "IOM"
        name = ", ".join(self._group_names[g] for g in groups)
        self.fig.suptitle(name)
        return "sheet_methods_contributing_integrated_"+"_".join(groups)

    @render_plot
    def sheet_methods_contributing_integrated_old(self, data, *groups):
        if not groups:
            groups = None

        for i, sheet in enumerate(IceSheet):
            self.ax = plt.subplot(230 + i + 1)
            self._sheet_methods_contributing_integrated_old(data, sheet, groups)

        if groups is None:
            groups = "RA", "GMB", "IOM"
        name = ", ".join(self._group_names[g] for g in groups)
        self.fig.suptitle(name)
        return "sheet_methods_contributing_integrated_" + "_".join(groups)

    def _sheet_methods_contributing_integrated(self,
        data: WorkingMassRateCollection, sheet: IceSheet, groups=None,
        yticks=True):

        # get default groups set
        if groups is None:
            groups = "RA", "GMB", "IOM"

        # get legible title of sheet
        title = self._sheet_names[sheet]
        # get id name of sheet
        sheet_id = sheet.value

        # set constituent sheets if 'AIS' is selected
        if sheet == IceSheet.ais:
            sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais]
        # otherwise, just use specified sheet
        else: sheets = [sheet]

        # get all series for the required set of ice sheets & groups
        set_data = data.filter(basin_id=sheets, user_group=groups)
        for group in groups:
            # get all series for experiment group
            group_data = set_data.filter(user_group=group)
            min_t = group_data.min_time()
            max_t = group_data.max_time()

            mid_t = min_t + (max_t - min_t) / 2.

            # get primary & secondary color for group
            pcol = style.colours.primary[group]
            scol = style.colours.secondary[group]

            # if there is no ice-sheet combination, plot the contributing series
            if len(sheets) == 1:

                for series in group_data.integrate(offset=mid_t):
                    self.ax.plot(series.t, series.dm, ls='--', color=pcol)

            # create empty group for per-sheet averages of this group
            group_avgs = WorkingMassRateCollection()

            for sheet in sheets:
                # add average of all series for sheet & group to collection
                group_avgs.add_series(
                    group_data.filter(basin_id=sheet).average()
                )
            # sum the collection
            series = group_avgs.sum().integrate(offset=mid_t)
            # plot the integrated sum
            self.ax.plot(series.t, series.dm, color=pcol)
            self.ax.fill_between(
                series.t, series.dm-series.errs, series.dm+series.errs,
                color=scol, alpha=0.5
            )

        # get start & end time of common period
        com_t_min = set_data.concurrent_start()
        com_t_max = set_data.concurrent_stop()
        # plot v. lines to show period
        self.ax.axvline(com_t_min, ls='--', color='k')
        self.ax.axvline(com_t_max, ls='--', color='k')

        # set title & axis labels
        self.ax.set_ylabel("Mass Change (Gt)")
        self.ax.set_title(title)
        # set x- & y-axis limits
        self.ax.set_ylim(self._dm0, self._dm1)
        self.ax.set_xlim(self._time0, self._time1)

        return "sheet_methods_contributing_integrated_" + sheet_id, \
               {'frameon': False, 'loc': 3}

    def _sheet_methods_contributing_integrated_old(self,
        data: MassRateCollectionsManager, sheet: IceSheet, groups=None,
        yticks=True):
        if groups is None:
            groups = "RA", "GMB", "IOM"

        title = self._sheet_names[sheet]
        sheet_id = sheet.value

        groups_time = {g: [] for g in groups}
        groups_rate = {g: [] for g in groups}
        groups_errs = {g: [] for g in groups}

        t_min = {g: None for g in groups}
        t_max = {g: None for g in groups}

        if sheet == IceSheet.ais:
            sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais]
        else: sheets = [sheet]

        for sheet in sheets:
            sheet_rate = {g: [] for g in groups}
            sheet_errs = {g: [] for g in groups}
            sheet_time = {g: [] for g in groups}

            for series in data[sheet]:
                g = series.user_group
                if g not in groups:
                    continue

                t, dmdt, errs = chunk_rates(series)

                sheet_time[g].append(t)
                sheet_rate[g].append(dmdt)
                sheet_errs[g].append(errs)

                if t_min[g] is None or t_min[g] > series.min_time:
                    t_min[g] = series.min_time
                if t_max[g] is None or t_max[g] < series.max_time:
                    t_max[g] = series.max_time

            for g in groups:
                t, rate = ts_combine(sheet_time[g], sheet_rate[g])
                _, errs = ts_combine(sheet_time[g], sheet_errs[g], error=True)

                groups_time[g].append(t)
                groups_rate[g].append(rate)
                groups_errs[g].append(errs)

                if len(sheets) == 1:
                    pcol = style.colours.primary[g]
                    t_mid = t_min[g] + (t_max[g] - t_min[g]) / 2.

                    for i, t in enumerate(sheet_time[g]):
                        mass = np.cumsum(sheet_rate[g][i]) / 12.
                        mass = apply_offset(t, mass, t_mid)
                        self.ax.plot(t, mass, color=pcol, ls='--')

        for g in groups:
            if not groups_time[g]: continue

            if len(sheets) > 1:
                t, rate = sum_sheets(groups_time[g], groups_rate[g])
                _, errs = sum_sheets(groups_time[g], groups_errs[g])
            else:
                t = groups_time[g][0]
                rate = groups_rate[g][0]
                errs = groups_errs[g][0]

            t_mid = t_min[g] + (t_max[g] - t_min[g]) / 2.

            self.labels.append(
                self._group_names[g]
            )
            self.glyphs.append(
                self.group_glyph(g)
            )

            errs = np.cumsum(errs) / 12.
            mass = np.cumsum(rate) / 12.

            mass = apply_offset(t, mass, t_mid)

            pcol = style.colours.primary[g]
            scol = style.colours.secondary[g]

            self.ax.plot(t, mass, color=pcol)
            self.ax.fill_between(
                t, mass-errs, mass+errs,
                color=scol, alpha=.5
            )


        com_t_min = max(t_min.values())
        com_t_max = min(t_max.values())
        self.ax.axvline(com_t_min, ls='--', color='k')
        self.ax.axvline(com_t_max, ls='--', color='k')

        self.ax.set_ylabel("Mass Change (Gt)")
        self.ax.set_title(title)
        self.ax.set_ylim(self._dm0, self._dm1)
        self.ax.set_xlim(self._time0, self._time1)

        return "sheet_methods_contributing_integrated_"+sheet_id,\
               {'frameon': False, 'loc': 3}


    @render_plot_with_legend
    def sheet_methods_average_integrated(self, data, sheets=None, groups=None, offset_t=None):
        if sheets is None:
            sheets = IceSheet
        for i, sheet in enumerate(sheets):
            self.ax = plt.subplot(230+i+1)
            self._sheet_methods_average_integrated(data, sheet, groups,
                                                   offset_t, yticks=False)
        if groups is None:
            groups = "RA", "GMB", "IOM"
        self.glyphs = [self.group_glyph(g) for g in groups]
        self.labels = [self._group_names[g] for g in groups]

        self.ax = plt.subplot(236)
        self.ax.tick_params(axis='both', which='both',
                            left='off', right='off',
                            top='off', bottom='off',
                            labelleft='off', labelbottom='off')
        sheet_names = "_".join(s.value for s in sheets)

        return "sheet_methods_average_integrated_"+sheet_names, {'frameon': False}



    def _sheet_methods_average_integrated(self, data: MassRateCollectionsManager,
                                         sheet: IceSheet, groups=None, offset_t=None, yticks=True):
        if groups is None:
            groups = "RA", "GMB", "IOM"
        title = self._sheet_names[sheet]
        sheet_id = sheet.value

        groups_time = {g: [] for g in groups}
        groups_rate = {g: [] for g in groups}
        groups_errs = {g: [] for g in groups}
        groups_time["all"] = []
        groups_rate["all"] = []
        groups_errs["all"] = []

        t_min = {g: None for g in groups}
        t_max = {g: None for g in groups}
        t_min["all"] = None
        t_max["all"] = None

        if sheet == IceSheet.ais:
            sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais]
        else: sheets = [sheet]

        for sheet in sheets:
            sheet_rate = {g: [] for g in groups}
            sheet_errs = {g: [] for g in groups}
            sheet_time = {g: [] for g in groups}

            for series in data[sheet]:
                g = series.user_group
                if g not in groups:
                    continue

                t, dmdt, errs = chunk_rates(series)

                sheet_time[g].append(t)
                sheet_rate[g].append(dmdt)
                sheet_errs[g].append(errs)

                if t_min[g] is None or t_min[g] > series.min_time:
                    t_min[g] = series.min_time
                if t_max[g] is None or t_max[g] < series.max_time:
                    t_max[g] = series.max_time
                if t_min["all"] is None or t_min["all"] > series.min_time:
                    t_min["all"] = series.min_time
                if t_max["all"] is None or t_max["all"] < series.max_time:
                    t_max["all"] = series.max_time

            all_time = []
            all_rate = []
            all_errs = []
            for g in groups:
                t, rate = ts_combine(sheet_time[g], sheet_rate[g])
                _, errs = ts_combine(sheet_time[g], sheet_errs[g], error=True)

                groups_time[g].append(t)
                groups_rate[g].append(rate)
                groups_errs[g].append(errs)
                all_time.append(t)
                all_rate.append(rate)
                all_errs.append(errs)

            t, rate = ts_combine(all_time, all_rate)
            _, errs = ts_combine(all_time, all_errs, error=True)

            groups_time["all"].append(t)
            groups_rate["all"].append(rate)
            groups_errs["all"].append(errs)

        for g in chain(groups, ["all"]):

            if len(sheets) > 1:
                t, rate = sum_sheets(groups_time[g], groups_rate[g])
                _, errs = sum_sheets(groups_time[g], groups_errs[g])
            else:
                t = groups_time[g][0]
                rate = groups_rate[g][0]
                errs = groups_errs[g][0]

            if offset_t is None:
                t_mid = t_min[g] + (t_max[g] - t_min[g]) / 2.
            else: t_mid = offset_t

            self.labels.append(
                self._group_names[g]
            )
            self.glyphs.append(
                self.group_glyph(g)
            )

            errs = np.cumsum(errs) / 12.
            mass = np.cumsum(rate) / 12.

            mass = apply_offset(t, mass, t_mid)

            pcol = style.colours.primary[g]
            scol = style.colours.secondary[g]

            self.ax.plot(t, mass, color=pcol, lw=2)
            self.ax.fill_between(
                t, mass-errs, mass+errs,
                color=scol, alpha=.5
            )

        com_t_min = max(t_min.values())
        com_t_max = min(t_max.values())
        self.ax.axvline(com_t_min, ls='--', color='k')
        self.ax.axvline(com_t_max, ls='--', color='k')

        # self.ax.set_ylabel("Mass Change (Gt)", fontweight='bold')
        self.ax.set_title(title, fontweight='bold')
        self.ax.set_ylim(self._dm0, self._dm1)
        self.ax.set_xlim(self._time0, self._time1)
        self.ax.set_xticks([1995, 2005, 2015])

        if not yticks:
            self.ax.tick_params(axis='y', which='both', left='off',
                                labelleft='off', right='off')


        name = "sheet_methods_average_integrated_"
        if offset_t is not None:
            name += str(offset_t) + "_"
        return name+sheet_id,\
               {'frameon': False, 'loc': 3}

    @render_plot
    def sheet_methods_average_rates(self, data, *groups):
        for i, sheet in enumerate(IceSheet):
            self.ax = plt.subplot(230 + i + 1)
            self._sheet_methods_average_rates(data, sheet, groups)
        names = ", ".join(self._group_names[g] for g in groups)
        self.fig.suptitle(names)

        return "sheet_methods_average_rates_"+"_".join(groups)

    def _sheet_methods_average_rates(self, data: MassRateCollectionsManager,
                                    sheet: IceSheet, groups=None):
        title = self._sheet_names[sheet]
        sheet_id = sheet.value

        if groups is None:
            groups = "RA", "GMB", "IOM"
        groups_time = {g: [] for g in groups}
        groups_rate = {g: [] for g in groups}
        groups_errs = {g: [] for g in groups}
        t_min = {g: None for g in groups}
        t_max = {g: None for g in groups}


        if sheet == IceSheet.ais:
            sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais]
        else:
            sheets = [sheet]

        for sheet in sheets:
            sheet_rate = {g: [] for g in groups}
            sheet_errs = {g: [] for g in groups}
            sheet_time = {g: [] for g in groups}

            for series in data[sheet]:
                g = series.user_group
                if g not in groups:
                    continue

                t, dmdt, errs = chunk_rates(series)

                sheet_time[g].append(t)
                sheet_rate[g].append(dmdt)
                sheet_errs[g].append(errs)

                if t_min[g] is None or t_min[g] > series.min_time:
                    t_min[g] = series.min_time
                if t_max[g] is None or t_max[g] < series.max_time:
                    t_max[g] = series.max_time

            for g in groups:
                t, rate = ts_combine(sheet_time[g], sheet_rate[g])
                _, errs = ts_combine(sheet_time[g], sheet_errs[g], error=True)

                groups_time[g].append(t)
                groups_rate[g].append(rate)
                groups_errs[g].append(errs)

                if len(sheets) == 1:
                    pcol = style.colours.primary[g]

                    for i, t in enumerate(sheet_time[g]):
                        self.ax.plot(t, sheet_rate[g][i], color=pcol, ls='--')

        for g in groups:

            if len(sheets) > 1:
                t, rate = sum_sheets(groups_time[g], groups_rate[g])
                _, errs = sum_sheets(groups_time[g], groups_errs[g])
            else:
                t = groups_time[g][0]
                rate = groups_rate[g][0]
                errs = groups_errs[g][0]

            self.labels.append(
                self._group_names[g]
            )
            self.glyphs.append(
                self.group_glyph(g)
            )

            pcol = style.colours.primary[g]
            scol = style.colours.secondary[g]

            self.ax.plot(t, rate, color=pcol)
            self.ax.fill_between(
                t, rate-errs, rate+errs,
                color=scol, alpha=.5
            )
        com_t_min = max(t_min.values())
        com_t_max = min(t_max.values())
        self.ax.axvline(com_t_min, ls='--', color='k')
        self.ax.axvline(com_t_max, ls='--', color='k')

        self.ax.axhline(0, color='k')
        plt.ylabel("Mass Balance (Gt/yr)")
        plt.title(title)
        self.ax.set_ylim(self._dmdt0, self._dmdt1)
        self.ax.set_xlim(self._time0, self._time1)

        return "sheet_methods_average_rates_"+sheet_id,\
               {'frameon': False, 'loc': 3}

    #### NEW PLOTTING METHODS:

    @render_plot
    def group_rate_intracomparison(self, group_avgs: WorkingMassRateCollection,
                                   group_contribs: WorkingMassRateCollection, regions):
        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(230+i+1)

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
            self.ax.set_ylim(self._dmdt0, self._dmdt1)
            self.ax.set_xlim(self._time0, self._time1)

        self.fig.suptitle("dM/dt intracomparison")
        return "group_rate_intracomparison"

    @render_plot
    def group_mass_intracomparison(self, group_avgs: MassChangeCollection, group_contribs: MassChangeCollection,
                                   regions):
        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(230+i+1)

            avg = group_avgs.filter(basin_id=name).average()
            if avg is None:
                print(name)
                continue

            pcol = style.colours.primary[avg.user_group]
            scol = style.colours.secondary[avg.user_group]

            self.ax.plot(avg.t, avg.dm, color=pcol)
            self.ax.fill_between(
                avg.t, avg.dm - avg.errs, avg.dm + avg.errs,
                color=scol, alpha=.5
            )

            if len(sheets) == 1:
                for contrib in group_contribs.filter(basin_id=sheets):
                    self.ax.plot(contrib.t, contrib.dm, color=pcol, ls='--')

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
            self.ax.set_ylim(self._dm0, self._dm1)
            self.ax.set_xlim(self._time0, self._time1)

        self.fig.suptitle("dM intracomparison")
        return "group_mass_intracomparison"

    @render_plot
    def groups_rate_intercomparison(self, region_avgs: WorkingMassRateCollection, group_avgs: WorkingMassRateCollection,
                                    regions):

        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(230+i+1)

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
            self.ax.set_ylim(self._dmdt0, self._dmdt1)
            self.ax.set_xlim(self._time0, self._time1)

        self.fig.suptitle("dM/dt intercomparison")
        return "groups_rate_intercomparison"

    @render_plot
    def groups_mass_intercomparison(self, region_avgs: MassChangeCollection, group_avgs: MassChangeCollection,
                                    regions):
        for i, (name, sheets) in enumerate(regions.items()):
            self.ax = plt.subplot(230+i+1)

            x_avg = region_avgs.filter(basin_id=name).average()
            if x_avg is None:
                print(name)
                continue

            pcol = style.colours.primary["all"]
            scol = style.colours.secondary["all"]

            self.ax.plot(x_avg.t, x_avg.dm, color=pcol)
            self.ax.fill_between(
                x_avg.t, x_avg.dm - x_avg.errs, x_avg.dm + x_avg.errs,
                color=scol, alpha=.5
            )

            for g_avg in group_avgs.filter(basin_id=name):
                pcol = style.colours.primary[g_avg.user_group]
                scol = style.colours.secondary[g_avg.user_group]

                self.ax.plot(g_avg.t, g_avg.dm, color=pcol)
                self.ax.fill_between(
                    g_avg.t, g_avg.dm - g_avg.errs, g_avg.dm + g_avg.errs,
                    color=scol, alpha=.5
                )

            # get start & end time of common period
            com_t_min = group_avgs.concurrent_start()
            com_t_max = group_avgs.concurrent_stop()
            # plot v. lines to show period
            self.ax.axvline(com_t_min, ls='--', color='k')
            self.ax.axvline(com_t_max, ls='--', color='k')

            # set title & axis labels
            self.ax.set_ylabel("Mass Change (Gt)")
            self.ax.set_title(self._sheet_names[name])
            # set x- & y-axis limits
            self.ax.set_ylim(self._dm0, self._dm1)
            self.ax.set_xlim(self._time0, self._time1)

        self.fig.suptitle("dM intercomparison")
        return "groups_mass_intercomparison"

    @render_plot_with_legend
    def regions_mass_intercomparison(self, region_avgs: MassChangeCollection, *regions: Sequence[IceSheet]):
        pcols = cycle(["#531A59", "#1B8C6F", "#594508"])
        scols = cycle(["#9E58A5", "#4CA58F", "#D8B54D"])

        for region, pcol, scol in zip(regions, pcols, scols):
            avg = region_avgs.filter(basin_id=region).average()

            self.ax.plot(avg.t, avg.dm, color=pcol)
            self.ax.fill_between(
                avg.t, avg.dm-avg.errs, avg.dm+avg.errs,
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
        # set x- & y-axis limits
        self.ax.set_ylim(self._dm0, self._dm1)
        self.ax.set_xlim(self._time0, self._time1)

        return "regions_mass_intercomparison_"+"_".join(r.value for r in regions), {"frameon": False, "loc": 3}
from imbie2.data.user import UserData
from imbie2.const.basins import *
from imbie2.const import AverageMethod
from imbie2.model.managers import *
from imbie2.plot import Plotter

import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib import markers
import sys, os
from itertools import chain


def main():
    logging.basicConfig(level=logging.CRITICAL)
    # rate_mgr = MassRateCollectionsManager(2002., 2012.)
    # mass_mgr = MassChangeCollectionsManager(2002., 2012.)
    rate_mgr = MassRateCollectionsManager(2003., 2012.)
    mass_mgr = MassChangeCollectionsManager()

    if len(sys.argv) >= 2:
        root = sys.argv[1]
    else: root = None

    if len(sys.argv) >= 3:
        out = os.path.expanduser(sys.argv[2])
    else: out = os.getcwd()

    names = set()
    fullnames = set()
    for user in UserData.find(root):
        if user.name in ["BDVGI", "mtalpe", "vhelm"]: #
            # BDVGI: no end times
            # mtalpe: huge values
            # vhelm: duplicate times
            continue
        fullname = user.forename + " " + user.lastname

        for series in user.rate_data():
            if series is None: continue
            rate_mgr.add_series(series)
            names.add(series.user)

            fullnames.add(fullname)

        for series in user.mass_data():
            if series is None: continue
            mass_mgr.add_series(series)
            names.add(series.user)

            fullnames.add(fullname)

        vals = []
        if not sheets: continue

    import matplotlib as mpl
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.size'] = 18
    plotter = Plotter()
    print("\n".join(fullnames))

    # for sheet in IceSheet:
    #     plotter.split_time_bars(rate_mgr, mass_mgr, sheet, list(names))
    # main_sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais, IceSheet.gris]
    # plotter.sheets_time_bars(rate_mgr, main_sheets, list(names))

    rate_mgr.merge()
    plotter.sheets_error_bars(rate_mgr, 2003., 2012.)
    # plotter.sheet_methods_average_integrated(rate_mgr, offset_t=2003)

    sys.exit(0)

    ais_sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais]
    for meth in AverageMethod:
        if meth == AverageMethod.split_altimetry:
            continue
        plotter.mass_series_continent_sheets(
            rate_mgr, [[s] for s in ais_sheets], ["APIS", "EAIS", "WAIS"],
            avg_method=meth
        )

        plotter.mass_series_continent_sheets(
            rate_mgr, [ais_sheets, [IceSheet.gris], ais_sheets+[IceSheet.gris]],
            ["AIS", "GrIS", "AIS+GrIS"], avg_method=meth
        )
    plotter.mass_series_continent_sheets(
        rate_mgr, [[s] for s in ais_sheets], ["APIS", "EAIS", "WAIS"],
        min_t=2003, max_t=2012
    )

    plotter.mass_series_continent_sheets(
        rate_mgr, [ais_sheets, [IceSheet.gris], ais_sheets+[IceSheet.gris]],
        ["AIS", "GrIS", "AIS+GrIS"], min_t=2003, max_t=2012
    )



    for group in "RA", "GMB", "IOM":
        plotter.box_coverage(rate_mgr, group)
        plotter.sheet_methods_contributing_integrated(rate_mgr, group)
        plotter.sheet_methods_average_rates(rate_mgr, group)
    # for sheet in IceSheet:
    #     plotter.sheet_methods_contributing_integrated(rate_mgr, sheet)
    #     plotter.sheet_methods_average_integrated(rate_mgr, sheet, offset_t=2003)
    #     plotter.sheet_methods_average_integrated(rate_mgr, sheet)
    #     plotter.sheet_methods_average_rates(rate_mgr, sheet)
    #     plotter.rate_series_sheet_groups(rate_mgr, sheet)
        # plotter.box_coverage(rate_mgr, sheet)
    plotter.basin_errors(ZwallyBasin, rate_mgr, "Zwally")
    plotter.basin_errors(RignotBasin, rate_mgr, "Rignot")




    # for basin in rate_mgr:
    #     for series in basin:
    #         print(series.basin_id.value, series.user_group, series.mean, '+/-', series.sigma, sep='\t')

    # for sheet in IceSheet:
    #     if sheet == IceSheet.ais: continue
    #
    #     plotter.sheet_scatter(sheet, ZwallyBasin, mass_mgr)
    #     plotter.sheet_scatter(sheet, RignotBasin, mass_mgr)

def main_old():
    # logging.captureWarnings(True)
    logging.basicConfig(level=logging.CRITICAL)

    rate_mgr = MassRateCollectionsManager()
    mass_mgr = MassChangeCollectionsManager()

    if len(sys.argv) >= 2:
        root = sys.argv[1]
    else: root = None

    if len(sys.argv) >= 3:
        out = os.path.expanduser(sys.argv[2])
    else: out = os.getcwd()

    print('            USERNAME', 'GROUP', 'MASS', 'RATE', sep='\t')
    print('--------------------', '-----', '----', '----', sep='\t')
    for user in UserData.find(root):
        name = user.lastname
        while len(name) < 20:
            name = ' ' + name
        print(name, user.group, user.has_mass_data, user.has_rate_data, sep='\t')
        if user.name in ["jmouginot", "BDVGI"]: #
            continue

        # print(user.name, user.group)
        # print("dM/dt data:", user.has_rate_data)
        for series in user.rate_data():
            rate_mgr.add_series(series)

            # items = [
            #     user.name, user.group, series.basin_group.value,
            #     series.basin_id.value, series.min_time, series.max_time,
            #     len(series)
            # ]
            # line = ','.join(str(i) for i in items)
            # print(line)
        # print("   dM data:", user.has_mass_data)
        for series in user.mass_data():
            if series is None: continue

            # items = [
            #     user.name, user.group, series.basin_group.value,
            #     series.basin_id.value, series.min_time, series.max_time,
            #     len(series)
            # ]
            # line = ','.join(str(i) for i in items)
            # print(line)
            mass_mgr.add_series(series)
        #     if series is None: continue
        #     print('\t', series.min_time, series.max_time, series.basin_id.value, len(series))
    print('mass:', len(mass_mgr))
    print('rate:', len(rate_mgr))

    colours = {
        'RA': 'red',
        'GMB': 'green',
        'IOM': 'blue'
    }
    grp_names = {
        'RA': 'Altimetry',
        'GMB': 'Gravimetry',
        'IOM': 'Mass Budget'
    }
    names = {
        IceSheet.apis: "Antarctic Peninsula",
        IceSheet.wais: "West Antarctica",
        IceSheet.eais: "East Antarctica",
        IceSheet.gris: "Greenland",
        IceSheet.ais: "Antarctica"
    }

    for basin in IceSheet:
        try:
            mass_collection = mass_mgr[basin]
        except KeyError: continue

        min_t = 1990
        max_t = 2020
        ax = plt.gca()

        users = []
        starts = []
        lengths = []
        groups = []
        order = []
        i = 0
        for series in mass_collection:
            if series.computed: continue

            i += 1
            order.append(i / 5.)
            starts.append(series.min_time)
            lengths.append(series.max_time - series.min_time)
            users.append(series.user)
            groups.append(colours[series.user_group])

        ax.barh(order, lengths, height=.1, left=starts, color=groups, tick_label=users)
        ax.set_xlim(min_t, max_t)
        ax.set_ylim(-.4, order[-1]+.4)
        # ax.minorticks_on()
        ax.xaxis.grid(b=True, which='major', linestyle='--')

        # create legend
        items = []
        labels = []
        for group in 'RA', 'GMB', 'IOM':
            name = grp_names[group]
            col = colours[group]

            items.append(
                patches.Patch(color=col, label=name)
            )
            labels.append(name)

        leg = plt.legend(items, labels, loc='upper left')

        plt.title('dM data time coverage: '+names[basin])
        # plt.show()
        fname = basin.value+'_times.png'
        fpath = os.path.join(out, fname)
        plt.savefig(fpath, bbox_inches='tight')
        print(fpath)
        plt.gca().clear()

    mkrs = [
        'o', 'v', '8', 'D', 's', 'p', 'h', '^', '*', '>', '<',
        'H', 'd', '+', 'x', '.', '|', '_', '1', '2', '3', '4',
        ',', markers.TICKLEFT, markers.TICKRIGHT, markers.TICKUP
    ]

    for sheet in IceSheet:
        if sheet == IceSheet.ais: continue

        titles = ["Rignot Basins", "Zwally Basins"]
        for grp_name, basinSet in zip(titles, [RignotBasin, ZwallyBasin]):
            ax = plt.gca()

            i = 0
            _any = False
            items = []
            labels = []
            for basin in basinSet.sheet(sheet):
                try:
                    collection = mass_mgr[basin]
                except KeyError: continue
                s = mkrs[i]
                i += 1

                p = None
                for series in collection:
                    if series.computed: continue
                    c = colours[series.user_group]
                    p = ax.scatter(series.t, series.dM,
                                   c=c, marker=s)

                if p is not None:
                    _any = True
                    icon = mlines.Line2D([],[],color='black', marker=s)
                    items.append(icon)
                    labels.append(basin.value)
            if _any:
                # create legend
                for group in 'RA', 'GMB', 'IOM':
                    name = grp_names[group]
                    col = colours[group]

                    items.append(
                        patches.Patch(color=col, label=name)
                    )
                    labels.append(name)

                plt.legend(items, labels, frameon=False, prop={'size': 8})
                ax.grid()
                ax.axhline(0, color='black')
                ax.set_xlim(2000, 2020)
                plt.ylabel('dM (Gt)')
                title = names[sheet] + ': ' + grp_name
                plt.title(title)
                # plt.show()

                fname = sheet.value+'_'+grp_name+'.png'
                fpath = os.path.join(out, fname)
                plt.savefig(fpath, bbox_inches='tight')
                print(fpath)
                ax.clear()

    for basin in IceSheet: #chain(IceSheet, RignotBasin, ZwallyBasin):
        try:
            rate_collection = rate_mgr[basin]
        except KeyError: continue

        ax = plt.gca()
        min_t = 1990
        max_t = 2020
        min_r = None
        max_r = None

        for series in rate_collection:
            if series.user in ["Talpe", ]: continue

            col = colours[series.user_group]

            r_len = series.max_rate - series.min_rate
            r_pos = series.min_rate
            t_len = series.max_time - series.min_time
            t_pos = series.min_time

            if r_len == 0:
                r_pos -= series.dMdt_err[0]
                r_len += series.dMdt_err[0]

            if min_t is None or min_t > t_pos:
                min_t = t_pos
            if max_t is None or max_t < t_pos + t_len:
                max_t = t_pos + t_len
            if min_r is None or min_r > r_pos:
                min_r = r_pos
            if max_r is None or max_r < r_pos + r_len:
                max_r = r_pos + r_len

            if series.computed:
                rect = patches.Rectangle(
                    (t_pos, r_pos),
                    t_len, r_len,
                    edgecolor=col, hatch='\\/',
                    fill=None
                )
            else:
                rect = patches.Rectangle(
                    (t_pos, r_pos),
                    t_len, r_len,
                    facecolor=col, alpha=.4
                )
            ax.add_patch(rect)
        # create legend
        items = []
        labels = []
        for group in 'RA', 'GMB', 'IOM':
            name = grp_names[group]
            col = colours[group]

            items.append(
                patches.Patch(color=col, label=name, alpha=.5)
            )
            labels.append(name)

        leg = plt.legend(items, labels, loc='lower left', frameon=False)

        r_range = abs(max_r - min_r)
        r_pad = .05 * r_range
        t_range = abs(max_t - min_t)
        t_pad = .01 * t_range

        title = names.get(basin, basin.value)
        plt.title(title+'\n')
        plt.ylabel('dM/dt (Gt/yr)')
        ax.set_xlim([min_t-t_pad, max_t+t_pad])
        ax.set_ylim([min_r-r_pad, max_r+r_pad])

        ax.xaxis.tick_top()
        ax.axhline(0, color='black')

        fname = "{}.png".format(basin.value)
        fpath = os.path.join(out, fname)
        plt.savefig(fpath, bbox_inches='tight')
        print(fpath)
        # plt.show()
        ax.clear()

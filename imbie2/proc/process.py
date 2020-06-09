from collections import OrderedDict
from itertools import product
import os
import shutil
from typing import Union, Sequence
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from imbie2.proc.sum_basins import sum_basins
from imbie2.conf import ImbieConfig
from imbie2.const.basins import IceSheet, BasinGroup
from imbie2.const.error_methods import ErrorMethod
from imbie2.model.collections import WorkingMassRateCollection, MassChangeCollection, \
                                     MassRateCollection
from imbie2.plot.plotter import Plotter
from imbie2.plot import style
from imbie2.table.tables import MeanErrorsTable, TimeCoverageTable, BasinsTable, \
                                RegionAveragesTable, RegionGroupAveragesTable
from imbie2.proc.compare_windows import compare_windows
from imbie2.util.count_tolerance import count_tolerance
from imbie2.util.functions import ts2m, match, move_av
from imbie2.util.discharge import calculate_discharge
from imbie2.model.series import WorkingMassRateDataSeries, MassChangeDataSeries

def prepare_collection(collection: Union[MassRateCollection, MassChangeCollection], config: ImbieConfig):
    if isinstance(collection, MassRateCollection):
        # normalise dM/dt data
        return collection.chunk_series()
    elif isinstance(collection, MassChangeCollection):
        return collection.to_dmdt(
                truncate=config.truncate_dmdt, window=config.dmdt_window, method=config.dmdt_method
        )
    else:
        raise TypeError("Expected dM or dM/dt collection")

def process(input_data: Sequence[Union[MassRateCollection, MassChangeCollection]], config: ImbieConfig):

    groups = ["RA", "GMB", "IOM"]
    if config.include_la:
        groups.append("LA")
    for g in config.methods_skip:
        groups.remove(g)

    # find output directory
    output_path = os.path.expanduser(config.output_path)

    # check if it exists, clear it if not empty (or abort)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.listdir(output_path):
        msg = "WARNING: directory \"%s\" is not empty, contents will be deleted. Proceed? (Y/n): " % output_path
        choice = input(msg)
        if (not choice.lower() == 'y') and choice:
            print("Processor cancelled by user.")
            return
        shutil.rmtree(output_path)

    sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais, IceSheet.gris]
    regions = OrderedDict([
        (IceSheet.eais, [IceSheet.eais]),
        (IceSheet.apis, [IceSheet.apis]),
        (IceSheet.wais, [IceSheet.wais]),
        (IceSheet.ais, [IceSheet.apis, IceSheet.eais, IceSheet.wais]),
        (IceSheet.gris, [IceSheet.gris]),
        (IceSheet.all, [IceSheet.apis, IceSheet.eais, IceSheet.wais, IceSheet.gris])
    ])
    offset = config.align_date

    rate_data = WorkingMassRateCollection()
    for collection in input_data:
        collection = prepare_collection(collection, config)
        for series in collection:
            # check if there's already a series for this user & location
            existing = rate_data.filter(
                user_group=series.user_group, user=series.user, basin_id=series.basin_id, basin_group=series.basin_group
            )
            if not existing:
                rate_data.add_series(series)

    for series in rate_data.filter(user_group='RA'):
        print(series.user, series.basin_id, series.t.max(), 'dM' if series.computed else 'dM/dt')

    # find users who have provided a full ice sheet of basin data, but no ice sheet series.
    sum_basins(rate_data, sheets)

    # keep copies of zwally/rignot data before merging them
    zwally_data = rate_data.filter(basin_group=BasinGroup.zwally)
    rignot_data = rate_data.filter(basin_group=BasinGroup.rignot)

    # merge zwally/rignot
    rate_data.merge()

    mass_data = rate_data.integrate(offset=offset)

    # create empty collections for storing outputs
    groups_sheets_rate = WorkingMassRateCollection()
    groups_sheets_mass = MassChangeCollection()

    groups_regions_rate = WorkingMassRateCollection()
    groups_regions_mass = MassChangeCollection()

    sheets_rate = WorkingMassRateCollection()
    sheets_mass = MassChangeCollection()

    regions_rate = WorkingMassRateCollection()
    regions_mass = MassChangeCollection()

    for outlier in config.users_mark:
        data = rate_data.filter(user=outlier)
        for series in data:
            for t, dmdt, e in zip(series.t, series.dmdt, series.errs):
                print(outlier, series.basin_id, t, dmdt, e)

    if config.reduce_window is not None:
        assert config.reduce_window > 0
        rate_data = rate_data.reduce(config.reduce_window)

    for group in groups:
        for sheet in sheets:
            print("computing", group, "average for", sheet.value, end="... ")

            new_series = rate_data.filter(
                user_group=group, basin_id=sheet
            ).average(
                mode=config.combine_method,
                error_mode=config.group_avg_errors_method,
                nsigma=config.average_nsigma
            )
            if new_series is None:
                continue

            groups_sheets_rate.add_series(new_series)
            groups_sheets_mass.add_series(
                new_series.integrate(offset=offset)
            )
            print("done.")
        for region, sheets in regions.items():
            print("computing", group, "average for", region.value, end="... ")

            region_rate = groups_sheets_rate.filter(
                user_group=group, basin_id=sheets
            ).sum(error_method=config.sum_errors_method)
            if region_rate is None:
                continue

            region_rate.basin_id = region
            region_mass = region_rate.integrate(offset=offset)

            groups_regions_rate.add_series(region_rate)
            groups_regions_mass.add_series(region_mass)
            print("done.")

    output_path = os.path.expanduser(config.output_path)
    for sheet in sheets:
        print("computing inter-group average for", sheet.value, end="... ")

        sheet_rate_avg = groups_sheets_rate.filter(
            basin_id=sheet
        ).average(
            mode=config.combine_method,
            error_mode=config.sheet_avg_errors_method,
            nsigma=config.average_nsigma
        )
        if sheet_rate_avg is None:
            continue

        sheets_rate.add_series(sheet_rate_avg)
        sheets_mass.add_series(
            sheet_rate_avg.integrate(offset=offset)
        )
        print("done.")

    # compute region figures
    for region, sheets in regions.items():
        print("computing inter-group average for", region.value, end="... ")

        region_rate = sheets_rate.filter(
            basin_id=sheets
        ).sum(error_method=config.sum_errors_method)
        if region_rate is None:
            continue

        region_rate.basin_id = region

        regions_rate.add_series(region_rate)
        regions_mass.add_series(
            region_rate.integrate(offset=offset)
        )
        print("done.")

    # print tables
    output_path = os.path.expanduser(config.output_path)

    met = MeanErrorsTable(rate_data, style=config.table_format)
    filename = os.path.join(output_path, "mean_errors."+met.default_extension())

    print("writing table:", filename)
    met.write(filename)

    btz = BasinsTable(zwally_data, BasinGroup.zwally, style=config.table_format)
    filename = os.path.join(output_path, "zwally_basins."+btz.default_extension())

    print("writing table:", filename)
    btz.write(filename)

    btr = BasinsTable(rignot_data, BasinGroup.rignot, style=config.table_format)
    filename = os.path.join(output_path, "rignot_basins." + btr.default_extension())

    print("writing table:", filename)
    btr.write(filename)

    rat = RegionAveragesTable(
        regions_rate, list(regions.keys()),
        (1992, 2011), (1992, 2000), (1993, 2003), (2000, 2011), (2005, 2010), (2010, 2017), (1992, 2017),
        (1992, 1997), (1997, 2002), (2002, 2007), (2007, 2012), (2012, 2017),
        style=config.table_format
    )
    filename = os.path.join(output_path, "region_window_averages." + rat.default_extension())

    print("writing table:", filename)
    rat.write(filename)

    rgt = RegionGroupAveragesTable(
        groups_regions_rate, regions_rate, list(regions.keys()), 2005, 2015, groups, style=config.table_format
    )
    filename = os.path.join(output_path, "region_group_window_averages."+rgt.default_extension())

    print("writing table:", filename)
    rgt.write(filename)

    for group in groups:
        tct = TimeCoverageTable(rate_data.filter(user_group=group), style=config.table_format)
        filename = os.path.join(output_path, "time_coverage_" + group + "." + tct.default_extension())

        print("writing table:", filename)
        tct.write(filename)

    # draw plots
    plotter = Plotter(
        filetype=config.plot_format,
        path=output_path,
        limits=True
    )
    if len(input_data) == 2:
        from functools import partial
        prepare = partial(prepare_collection, config=config)
        data_a, data_b = map(prepare, input_data)
        for sheet in sheets:
            for group in groups:
                data_a_sel = data_a.filter(user_group=group, basin_id=sheet, user='Shepherd').window_cropped()
                data_b_sel = data_b.filter(user_group=group, basin_id=sheet, user='Shepherd').window_cropped()

                name = "%s_%s" % (group, sheet.value)
                plotter.named_dmdt_comparison_plot(data_a_sel, data_b_sel, name)

    # rignot/zwally comparison
    for sheet in sheets:
        plotter.rignot_zwally_comparison(
            rignot_data+zwally_data, [sheet]
        )
    # error bars (IMBIE1 style plot)
    window = config.bar_plot_min_time, config.bar_plot_max_time
    plotter.sheets_error_bars(
        groups_regions_rate.window_cropped(), regions_rate, groups, regions, window=window
    )
    plotter.sheets_error_bars(
        groups_regions_rate.window_cropped(), regions_rate, groups, regions,
        window=window, ylabels=True, suffix="labeled"
    )

    align_dm = offset is None
    # intracomparisons
    for group in groups:
        plotter.group_rate_boxes(
            rate_data.filter(user_group=group), {s: s for s in sheets}, suffix=group
        )
        plotter.group_rate_intracomparison(
            groups_regions_rate.filter(user_group=group).window_cropped().smooth(config.plot_smooth_window),
            rate_data.filter(user_group=group).window_cropped().smooth(config.plot_smooth_window),
            regions, suffix=group, mark=config.users_mark
        )
        plotter.group_mass_intracomparison(
            groups_regions_mass.filter(user_group=group),
            mass_data.filter(user_group=group), regions, suffix=group,
            mark=config.users_mark, align=align_dm
        )
        for sheet in sheets:
            plotter.named_dmdt_group_plot(
                sheet, group, rate_data.filter(user_group=group, basin_id=sheet).window_cropped(),
                groups_regions_rate.filter(user_group=group, basin_id=sheet).window_cropped().first()
            )
            plotter.named_dm_group_plot(
                sheet, group, mass_data.filter(user_group=group, basin_id=sheet),
                basis=groups_regions_mass.filter(user_group=group, basin_id=sheet).first()
            )
    # intercomparisons
    for _id, region in regions.items():
        reg = {_id: region}

        plotter.groups_rate_intercomparison(
            regions_rate.window_cropped().smooth(config.plot_smooth_window),
            groups_regions_rate.smooth(config.plot_smooth_window), reg
        )
        plotter.groups_mass_intercomparison(
            regions_mass, groups_regions_mass, reg, align=align_dm
        )
    # region comparisons
    ais_regions = [IceSheet.eais, IceSheet.wais, IceSheet.apis]
    all_regions = [IceSheet.ais, IceSheet.gris, IceSheet.all]

    plotter.regions_mass_intercomparison(
        regions_mass, *sheets
    )
    plotter.regions_mass_intercomparison(
        regions_mass, *ais_regions
    )
    plotter.regions_mass_intercomparison(
        regions_mass, *all_regions
    )

    if not config.export_data:
        return

    # write data to files
    for region in regions:
        data = regions_rate.filter(basin_id=region).first()
        fname = os.path.join(output_path, region.value+".csv")

        print("exporting data:", fname, end="... ")
        with open(fname, 'w') as f:
            for line in zip(data.t, data.dmdt, data.errs):
                line = ", ".join(map(str, line)) + "\n"
                f.write(line)
        print("done.")

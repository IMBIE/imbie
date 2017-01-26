from imbie2.const.basins import IceSheet
from imbie2.model.collections import WorkingMassRateCollection, MassChangeCollection, MassRateCollection
from imbie2.plot.plotter import Plotter
from imbie2.plot.table import MeanErrorsTable, TimeCoverageTable

from collections import OrderedDict


def process(input_data: MassRateCollection):

    groups = "RA", "GMB", "IOM"
    sheets = [IceSheet.apis, IceSheet.eais, IceSheet.wais, IceSheet.gris]
    regions = OrderedDict([
        (IceSheet.eais, [IceSheet.eais]),
        (IceSheet.apis, [IceSheet.apis]),
        (IceSheet.wais, [IceSheet.wais]),
        (IceSheet.ais, [IceSheet.apis, IceSheet.eais, IceSheet.wais]),
        (IceSheet.gris, [IceSheet.gris]),
        (IceSheet.all, [IceSheet.apis, IceSheet.eais, IceSheet.wais, IceSheet.gris])
    ])
    offset = 2003.

    rate_data = input_data.chunk_series()
    rate_data.merge()
    mass_data = rate_data.integrate(offset=offset)

    met = MeanErrorsTable(rate_data)
    print(met)

    for group in groups:
        tct = TimeCoverageTable(rate_data.filter(user_group=group))
        print(tct)

    groups_sheets_rate = WorkingMassRateCollection()
    groups_sheets_mass = MassChangeCollection()

    groups_regions_rate = WorkingMassRateCollection()
    groups_regions_mass = MassChangeCollection()

    sheets_rate = WorkingMassRateCollection()
    sheets_mass = MassChangeCollection()

    regions_rate = WorkingMassRateCollection()
    regions_mass = MassChangeCollection()

    for group in groups:
        for sheet in sheets:
            new_series = rate_data.filter(
                user_group=group, basin_id=sheet
            ).average()
            if new_series is None:
                continue

            groups_sheets_rate.add_series(new_series)
            groups_sheets_mass.add_series(
                new_series.integrate(offset=offset)
            )

        for region, sheets in regions.items():
            region_rate = groups_sheets_rate.filter(
                user_group=group, basin_id=sheets
            ).sum()
            if region_rate is None:
                continue

            region_rate.basin_id = region
            region_mass = region_rate.integrate(offset=offset)

            groups_regions_rate.add_series(region_rate)
            groups_regions_mass.add_series(region_mass)

    for sheet in sheets:
        sheet_rate_avg = groups_sheets_rate.filter(basin_id=sheet).average()
        if sheet_rate_avg is None:
            continue

        sheets_rate.add_series(sheet_rate_avg)
        sheets_mass.add_series(
            sheet_rate_avg.integrate(offset=offset)
        )

    # compute region figures
    for region, sheets in regions.items():
        region_rate = sheets_rate.filter(
            basin_id=sheets
        ).sum()
        if region_rate is None:
            continue

        region_rate.basin_id = region

        regions_rate.add_series(region_rate)
        regions_mass.add_series(
            region_rate.integrate(offset=offset)
        )



    # draw plots
    plotter = Plotter()
    # intracomparisons
    for group in groups:
        plotter.group_rate_intracomparison(
            groups_regions_rate.filter(user_group=group),
            rate_data.filter(user_group=group), regions
        )
        plotter.group_mass_intracomparison(
            groups_regions_mass.filter(user_group=group),
            mass_data.filter(user_group=group), regions
        )
    # intercomparisons
    plotter.groups_rate_intercomparison(
        regions_rate, groups_regions_rate, regions
    )
    plotter.groups_mass_intercomparison(
        regions_mass, groups_regions_mass, regions
    )
    # region comparisons
    ais_regions = [IceSheet.eais, IceSheet.wais, IceSheet.apis]
    all_regions = [IceSheet.ais, IceSheet.gris, IceSheet.all]

    plotter.regions_mass_intercomparison(
        regions_mass, *ais_regions
    )
    plotter.regions_mass_intercomparison(
        regions_mass, *all_regions
    )

# class Plotter:
#     def __init__(self):
#
#     def group_rate_intracomparison(self, group_avgs: WorkingMassRateCollection,
#                                    group_contribs: WorkingMassRateCollection, **regions):
#
#         for i, name, sheets in enumerate(regions.items()):
#             avg = group_avgs.filter(basin_id=name).average()
#             pcol = style.colours.primary[avg.user_group]
#             scol = style.colours.secondary[avg.user_group]
#
#             self.ax.plot(avg.t, avg.dmdt, color=pcol)
#             self.ax.fill_between(
#                 avg.t, avg.dmdt-avg.errs, avg.dmdt+avg.errs,
#                 color=scol, alpha=.5
#             )
#
#             if len(sheets) > 1:
#                 continue
#
#             for contrib in group_contribs.filter(basin_id=sheets):
#                 self.ax.plot(contrib.t, contrib.dmdt, color=pcol, ls='--')
#
#     def group_mass_intracomparison(self, group_avgs: MassChangeCollection, group_contribs: MassChangeCollection,
#                                    **regions):
#         for i, name, sheets in enumerate(regions.items()):
#             avg = group_avgs.filter(basin_id=name).average()
#             pcol = style.colours.primary[avg.user_group]
#             scol = style.colours.secondary[avg.user_group]
#
#             self.ax.plot(avg.t, avg.dm, color=pcol)
#             self.ax.fill_between(
#                 avg.t, avg.dm-avg.errs, avg.dm+avg.errs,
#                 color=scol, alpha=.5
#             )
#
#             if len(sheets) > 1:
#                 continue
#
#             for contrib in group_contribs.filter(basin_id=sheets):
#                 self.ax.plot(contrib.t, contrib.dm, color=pcol, ls='--')
#
#
#     def groups_rate_intercomparison(self, region_avgs: WorkingMassRateCollection, group_avgs: WorkingMassRateCollection,
#                                     **regions):
#
#         for i, name, sheets in enumerate(regions.items()):
#             x_avg = region_avgs.filter(basin_id=name).average()
#             pcol = style.colours.primary["all"]
#             scol = style.colours.secondary["all"]
#
#             self.ax.plot(x_avg.t, x_avg.dmdt, color=pcol)
#             self.ax.fill_between(
#                 x_avg.t, x_avg.dmdt-x_avg.errs, x_avg.dmdt+x_avg.errs,
#                 color=scol, alpha=.5
#             )
#
#             for g_avg in group_avgs.filter(basin_id=name):
#                 pcol = style.colours.primary[g_avg.user_group]
#                 scol = style.colours.secondary[g_avg.user_group]
#
#                 self.ax.plot(g_avg.t, g_avg.dmdt, color=pcol)
#                 self.ax.fill_between(
#                     g_avg.t, g_avg.dmdt-g_avg.errs, g_avg.dmdt+g_avg.errs,
#                     color=scol, alpha=.5
#                 )
#
#     def groups_mass_intercomparison(self, region_avgs: MassChangeCollection, group_avgs: MassChangeCollection, **regions):
#         for i, name, sheets in enumerate(regions.items()):
#             x_avg = region_avgs.filter(basin_id=name).average()
#             pcol = style.colours.primary["all"]
#             scol = style.colours.secondary["all"]
#
#             self.ax.plot(x_avg.t, x_avg.dm, color=pcol)
#             self.ax.fill_between(
#                 x_avg.t, x_avg.dm-x_avg.errs, x_avg.dm+x_avg.errs,
#                 color=scol, alpha=.5
#             )
#
#             for g_avg in group_avgs.filter(basin_id=name):
#                 pcol = style.colours.primary[g_avg.user_group]
#                 scol = style.colours.secondary[g_avg.user_group]
#
#                 self.ax.plot(g_avg.t, g_avg.dm, color=pcol)
#                 self.ax.fill_between(
#                     g_avg.t, g_avg.dm-g_avg.errs, g_avg.dm+g_avg.errs,
#                     color=scol, alpha=.5
#                 )
#
#     def regions_mass_intercomparison(self, region_avgs: MassChangeCollection, **region):
#         pass
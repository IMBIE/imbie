from collections import OrderedDict

from imbie2.conf import ImbieConfig
from imbie2.const.basins import IceSheet, BasinGroup
from imbie2.model.collections import WorkingMassRateCollection, MassChangeCollection
from imbie2.plot.plotter import Plotter
from imbie2.table.tables import MeanErrorsTable, TimeCoverageTable, BasinsTable


def process_mass(mass_data: MassChangeCollection, config: ImbieConfig):

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
    #
    # rate_data = input_data

    rate_data = mass_data.differentiate()

    groups_sheets_rate = WorkingMassRateCollection()
    groups_sheets_mass = MassChangeCollection()

    groups_regions_rate = WorkingMassRateCollection()
    groups_regions_mass = MassChangeCollection()

    sheets_rate = WorkingMassRateCollection()
    sheets_mass = MassChangeCollection()

    regions_rate = WorkingMassRateCollection()
    regions_mass = MassChangeCollection()

    mass_data.filter(user_group='GMB', basin_id=IceSheet.wais).savemat("gmb_wais")

    for group in groups:
        for sheet in sheets:
            new_series = mass_data.filter(
                user_group=group, basin_id=sheet, _max=5
            ).combine()
            if new_series is None:
                continue

            groups_sheets_rate.add_series(
                new_series.differentiate()
            )
            groups_sheets_mass.add_series(new_series)

        for region, sheets in regions.items():
            region_mass = groups_sheets_mass.filter(
                user_group=group, basin_id=sheets
            ).sum()
            if region_mass is None:
                continue

            region_mass.basin_id = region
            region_rate = region_mass.differentiate()

            groups_regions_rate.add_series(region_rate)
            groups_regions_mass.add_series(region_mass)

    for sheet in sheets:
        sheet_mass_avg = groups_sheets_mass.filter(basin_id=sheet, _max=5).combine()
        if sheet_mass_avg is None:
            continue

        sheets_rate.add_series(
            sheet_mass_avg.differentiate()
        )
        sheets_mass.add_series(sheet_mass_avg)

    # compute region figures
    for region, sheets in regions.items():
        region_mass = sheets_mass.filter(
            basin_id=sheets
        ).sum()
        if region_mass is None:
            continue

        region_mass.basin_id = region

        regions_rate.add_series(region_mass.differentiate())
        regions_mass.add_series(region_mass)

        # print tables

        met = MeanErrorsTable(rate_data)
        # f.write(met.get_html_string())
        print(met)

        btz = BasinsTable(rate_data, BasinGroup.zwally)
        # f.write(btz.get_html_string())
        print(btz)

        btr = BasinsTable(rate_data, BasinGroup.rignot)
        # f.write(btr.get_html_string())
        print(btr)

        for group in groups:
            tct = TimeCoverageTable(rate_data.filter(user_group=group))
            # f.write(tct.get_html_string())
            print(tct)

    # print tables

    met = MeanErrorsTable(rate_data)
    print(met)

    for group in groups:
        tct = TimeCoverageTable(rate_data.filter(user_group=group))
        print(tct)

    # draw plots
    #### IMBIE3 update: added config file to pass on plot defaults
    
    plotter = Plotter(filetype=config.plot_format, path=config.output_path, config=config)
    
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

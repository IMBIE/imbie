"""
CLI script for pre-processing dM data files
"""
import pandas as pd
import numpy as np
import argparse as ap
from typing import Iterator
import os
import shutil
import sys

from imbie2.conf import ImbieConfig
from imbie2.util.dm_to_dmdt import dm_to_dmdt
from imbie2.model.series import MassChangeDataSeries, WorkingMassRateDataSeries
from imbie2.model.collections import MassChangeCollection, WorkingMassRateCollection
from imbie2.const.basins import BasinGroup, IceSheet, ZwallyBasin, RignotBasin
from imbie2.const.error_methods import ErrorMethod
from imbie2.proc.sum_basins import sum_basins

ICE_SHEETS = [IceSheet.apis, IceSheet.eais, IceSheet.wais, IceSheet.gris]
DM_COLUMNS = [
    "name",
    "group",
    "basin_type",
    "sheet",
    "area",
    "obs_area",
    "year",
    "dm",
    "dm_sd",
]
DMDT_COLUMNS = [
    "group",
    "name",
    "sheet",
    "basin_type",
    "year",
    "dmdt",
    "dmdt_sd",
]


def create_parser(name: str, desc: str) -> ap.ArgumentParser:
    """
    create CLI argument parser for script
    """
    p = ap.ArgumentParser(name, description=desc)
    p.add_argument("config", type=str, help="path to IMBIE-2 config file")
    p.add_argument(
        "--overwrite", action="store_true", help="overwrite outputs without asking"
    )

    return p


def parse_file(fpath: str) -> MassChangeDataSeries:
    """
    reads a CSV file and returns collection of parsed data series
    """
    # read raw CSV data with pandas
    data = pd.read_csv(
        fpath,
        names=DM_COLUMNS,
        skipinitialspace=True,
        comment="#",
        converters={"group": str.upper, "sheet": str.lower, "basin_type": str.lower},
        usecols=np.arange(len(DM_COLUMNS)),
    )

    # get user name and experiment group
    name = data.name[0]
    group = data.group[0]

    # check validity of data
    assert np.all(
        data.group == group
    ), f"multiple experiment groups in same file: {fpath:r}"
    assert np.all(
        data.name == name
    ), f"multiple contributor names in same file: {fpath:r}"

    # create empty output collection
    out = MassChangeCollection()

    # get all basin types specified
    basin_types = np.unique(data.basin_type)

    for btype in basin_types:

        ok1 = np.flatnonzero(data.basin_type == btype)
        basins = np.unique(data.sheet[ok1])

        for basin in basins:
            ok2 = np.flatnonzero(
                np.logical_and(data.basin_type == btype, data.sheet == basin)
            )

            if IceSheet.is_valid(basin):
                basin_enum = IceSheet.get_basin(basin)
                if btype in ("rignot", "zwally"):
                    basin_type_enum = BasinGroup(btype)
                else:
                    basin_type_enum = BasinGroup.sheets
            else:
                basin_type_enum = BasinGroup(btype)
                enum_type = {
                    BasinGroup.rignot: RignotBasin,
                    BasinGroup.zwally: ZwallyBasin,
                }[basin_type_enum]

                basin_enum = enum_type.parse(basin)

            series = MassChangeDataSeries(
                name,
                group,
                group,
                basin_type_enum,
                basin_enum,
                data.area[ok2].values,
                data.year[ok2].values,
                data.obs_area[ok2].values,
                data.dm[ok2].values,
                data.dm_sd[ok2].values,
                interpolate=False,
            )
            out.add_series(series)

    return out


def ask_overwrite(path: str) -> bool:
    """
    ask for permission to overwrite contents of folder
    """
    while True:
        resp = input(f"directory {path} already exists, overwrite? [Y/n]")
        if not resp or resp[0].lower() == "y":
            return True
        elif resp[0].lower() == "n":
            return False
        print(f'unrecognised input {resp:r}, expected "yes" or "no"')


def main() -> None:
    """
    main process for script
    """
    desc = "tool for pre-processing IMBIE dM contributions to create dM/dt data"
    parser = create_parser("IMBIE-2 dM data pre-processor", desc)
    args = parser.parse_args()

    config = ImbieConfig(args.config)
    config.open()

    root = os.path.expanduser(config.input_path)

    if os.path.exists(config.output_path):
        if not args.overwrite and not ask_overwrite(config.output_path):
            sys.exit(-1)
        shutil.rmtree(config.output_path)
    os.makedirs(config.output_path, exist_ok=True)

    mass_data = MassChangeCollection()

    for fname in os.listdir(root):
        fpath = os.path.join(root, fname)

        if not os.path.isfile(fpath):
            continue

        try:
            mass_data += parse_file(fpath)
        except Exception as e:
            print(f"error reading {fpath}:")
            raise e
        print(f"loaded {fpath}")

    sum_basins(mass_data)

    rate_data = mass_data.to_dmdt(
        config.truncate_dmdt,
        config.dmdt_window,
        config.dmdt_method,
        config.dmdt_tapering,
        config.dmdt_monthly,
    )

    users = {s.user for s in rate_data}

    for user in users:
        user_series = rate_data.filter(user=user).merge_basin_types()
        user_sheets = user_series.filter(basin_id=ICE_SHEETS)

        if not user_sheets:
            print(f"no ice sheet data available for {user}")
            continue

        if not user_sheets.filter(basin_id=IceSheet.ais):
            existing_sheets = [s.basin_id for s in user_sheets]
            if (
                IceSheet.wais in existing_sheets
                and IceSheet.eais in existing_sheets
                and IceSheet.apis in existing_sheets
            ):
                print(f"creating AIS series for {user}")

                ais_series = user_sheets.filter(
                    basin_id=[IceSheet.wais, IceSheet.eais, IceSheet.apis]
                ).sum(error_method=ErrorMethod.rss)

                ais_series.basin_id = IceSheet.ais
                ais_series.basin_group = BasinGroup.sheets
                ais_series.user = user
                ais_series.aggregated = True

                user_sheets.add_series(ais_series)

        sheets = []

        for series in user_sheets:
            rows = {
                "group": np.array([series.user_group] * series.t.size),
                "name": np.array([series.user] * series.t.size),
                "sheet": np.array([series.basin_id.value] * series.t.size),
                "basin_type": np.array([series.basin_group.value] * series.t.size),
                "year": series.t,
                "dmdt": series.dmdt,
                "dmdt_sd": series.errs,
            }

            sheets.append(pd.DataFrame(rows, columns=DMDT_COLUMNS))

        out_data = pd.concat(sheets)

        name = series.user.replace("/", "_")
        opath = os.path.join(config.output_path, f"{name}.csv")
        out_data.to_csv(opath, columns=DMDT_COLUMNS, header=False, index=False)

        print(f"created: {opath}")

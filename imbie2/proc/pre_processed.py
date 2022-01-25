import os
import pandas as pd
import numpy as np
import argparse as ap
from typing import Iterator

from imbie2.model.collections import WorkingMassRateCollection
from imbie2.model.series import WorkingMassRateDataSeries
from imbie2.proc.process import process
from imbie2.const.basins import BasinGroup, IceSheet
from imbie2.conf import ImbieConfig

COLUMN_NAMES = [
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
    create CLI argument parser
    """
    p = ap.ArgumentParser(name, description=desc)
    p.add_argument("config", type=str, help="Path to an IMBIE configuration file")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed warnings"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite output directory without prompt",
    )

    return p


def parse_file(fpath: str) -> Iterator[WorkingMassRateDataSeries]:
    """
    yields all series from a file
    """
    # read raw CSV using pandas
    data = pd.read_csv(fpath, names=COLUMN_NAMES, skipinitialspace=True, comment="#")

    # get name and experiment group from first row
    name = data.name[0]
    group = data.group[0]

    # check that file contains consistent data
    assert np.all(data.group == group), (
        "Multiple experiment groups specified: %s" % fpath
    )
    assert np.all(data.name == name), "Multiple contributor names specified: %s" % fpath

    if group.lower() == "gravimetry":
        group = "GMB"
    elif group.lower().replace(" ","") == "massbudget":
        group = "IOM"
    elif group.lower() == "altimetry":
        group = "RA"

    # find all the ice sheets in the file
    sheets = np.unique(data.sheet)

    for sheet in sheets:
        # find all rows relating to this sheet
        ok = np.flatnonzero(data.sheet == sheet)

        # convert string values to enums
        sheet_id = IceSheet(sheet.lower())
        basin_types = data.basin_type[ok].values

        assert np.all(basin_types == basin_types[0]), "multiple basin types specified"

        basin_type = BasinGroup(basin_types[0])
        # get numeric values for data
        series_year = data.year[ok].values
        series_dmdt = data.dmdt[ok].values
        series_sd = data.dmdt_sd[ok].values

        # fill area params with zeros
        series_area = np.zeros_like(series_year)
        series_obs = np.zeros_like(series_year)

        # create data series and add it to the collection
        yield WorkingMassRateDataSeries(
            name,
            group,
            group,
            basin_type,
            sheet_id,
            series_area,
            series_year,
            series_obs,
            series_dmdt,
            series_sd,
        )


def main() -> None:
    """
    entry point for running IMBIE process with pre-processed data files
    """
    desc = "IMBIE-2 processor for pre-processed data files"
    parser = create_parser("IMBIE-2 pre-processed", desc)

    args = parser.parse_args()

    config = ImbieConfig(args.config)
    config.open()

    root = os.path.expanduser(config.input_path)

    # create empty collection for data
    collection = WorkingMassRateCollection()

    # find all files in input directory
    for fname in os.listdir(root):
        fpath = os.path.join(root, fname)

        try:
            sheets = []
            for series in parse_file(fpath):
                collection.add_series(series)

                sheets.append(series.basin_id.value)
                group = series.user_group
                name = series.user
        except Exception as e:
            print(f"error loading file {fpath}:")
            print(e)
        else:
            print(f"loaded {group}/{name}:", ", ".join(sheets))

    process([collection], config, args.overwrite)

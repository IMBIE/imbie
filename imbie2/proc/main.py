from imbie2.data.user import UserData
from imbie2.const.basins import *
from imbie2.const.error_codes import ErrorCode
from imbie2.model.managers import *
from imbie2.conf import ImbieConfig, ConfigError
from imbie2.version import __version__
from .process import process
# from .process_mass import process_mass

import logging
import os
import argparse

__doc__ = """
The IMBIE processor.

This program discovers and parses data from IMBIE contributions
in order to collate, process, and analyse dM and dM/dt time-series
for ice sheets in Antarctica and Greenland.

Options are configured via a configuration file, which must be
provided as an argument to the processor.
"""

def main():
    """
    main IMBIE process
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=str, help="Path to an IMBIE configuration file")
    cfg_path = parser.parse_args().config

    print("IMBIE processor v{}".format(__version__))

    if not os.path.exists(cfg_path):
        print("config file does not exist: {}".format(cfg_path))
        return ErrorCode.config_missing.value

    print("reading configuration", end="... ")
    # read the config file
    config = ImbieConfig(cfg_path)
    try:
        config.open()
    except ConfigError as e:
        print("error reading config file:")
        print(e)
        return ErrorCode.config_invalid.value

    print("done.")

    # set logging level (todo: add this to config?)
    logging.basicConfig(level=logging.CRITICAL)

    # create empty dM/dt and dM managers
    rate_mgr = MassRateCollectionsManager(config.start_date, config.stop_date)
    mass_mgr = MassChangeCollectionsManager(config.start_date, config.stop_date)

    # expand input directory
    root = os.path.expanduser(config.input_path)
    if not os.path.exists(root):
        print("input directory does not exist")
        return ErrorCode.input_path.value

    # create sets for usernames & full names
    names = set()
    fullnames = set()

    print("reading input data from", root, end="... ")
    # search input directory
    for user in UserData.find(root):
        if user.name in config.users_skip:
            continue
        fullname = user.forename + " " + user.lastname

        for series in user.rate_data():
            if series is None:
                continue

            rate_mgr.add_series(series)

            names.add(series.user)
            fullnames.add(fullname)

            line = ", ".join(str(s) for s in [fullname, series.user, series.basin_group, series.basin_id]) + "\n"
            logging.info(line)

        for series in user.mass_data(convert=False):
            if series is None:
                continue

            mass_mgr.add_series(series)

            names.add(series.user)
            fullnames.add(fullname)

        if not sheets:
            continue
    print("done.")

    if not rate_mgr:
        print("no data in input directory")
        return ErrorCode.no_data.value
    else:
        print(len(rate_mgr), "contributions read")

    # convert manager to collection
    rate_col = rate_mgr.as_collection()
    # process the data
    process(rate_col, config)

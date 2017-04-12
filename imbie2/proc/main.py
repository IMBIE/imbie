from imbie2.data.user import UserData
from imbie2.const.basins import *
from imbie2.model.managers import *
from imbie2.conf import ImbieConfig
from imbie2.version import __version__

import logging
import os
from .process import process
from .process_mass import process_mass


def main():
    """
    main IMBIE2 process
    """
    print("IMBIE processor v{}".format(__version__))

    print("reading configuration", end="... ")
    # read the config file
    config = ImbieConfig("config")
    config.open()
    print("done.")

    # set logging label (todo: add this to config?)
    logging.basicConfig(level=logging.CRITICAL)

    # create empty dM/dt and dM managers
    rate_mgr = MassRateCollectionsManager(config.start_date, config.stop_date)
    mass_mgr = MassChangeCollectionsManager(config.start_date, config.stop_date)

    # expand input directory
    root = os.path.expanduser(config.input_path)

    # create sets for usernames & full names
    names = set()
    fullnames = set()

    print("reading input data from", root, end="... ")
    # search input directory
    log = open("log.txt", 'w')
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

            log.write(", ".join(str(s) for s in [fullname, series.user, series.basin_group, series.basin_id]) + "\n")

        for series in user.mass_data(convert=False):
            if series is None:
                continue

            mass_mgr.add_series(series)

            names.add(series.user)
            fullnames.add(fullname)

        if not sheets:
            continue
    print("done.")
    log.close()

    # rate_col = mass_mgr.as_collection().differentiate() + rate_mgr.as_collection().chunk_series()

    # convert manager to collection
    rate_col = rate_mgr.as_collection()
    mass_col = mass_mgr.as_collection()
    # process the data

    # process_mass(mass_col, config)
    process(rate_col, config)

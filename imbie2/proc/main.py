from imbie2.data.user import UserData
from imbie2.const.basins import *
from imbie2.model.managers import *
from imbie2.conf import ImbieConfig

import logging
import os
from .process import process


def main():
    """
    main IMBIE2 process
    """
    # read the config file
    config = ImbieConfig("config")
    config.open()

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

        for series in user.mass_data(convert=False):
            if series is None:
                continue

            mass_mgr.add_series(series)

            names.add(series.user)
            fullnames.add(fullname)

        if not sheets:
            continue

    # rate_col = mass_mgr.as_collection().differentiate() + rate_mgr.as_collection().chunk_series()

    # convert manager to collection
    rate_col = rate_mgr.as_collection()
    # process the data
    process(rate_col, config)

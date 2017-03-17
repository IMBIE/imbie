from imbie2.data.user import UserData
from imbie2.const.basins import *
from imbie2.model.managers import *
from imbie2.conf import ImbieConfig

import logging
import os
from .process import process


def main():
    config = ImbieConfig("config")
    config.open()

    logging.basicConfig(level=logging.CRITICAL)

    rate_mgr = MassRateCollectionsManager(config.start_date, config.stop_date) # 2003., 2012.
    mass_mgr = MassChangeCollectionsManager(config.start_date, config.stop_date)

    root = os.path.expanduser(config.input_path)

    names = set()
    fullnames = set()

    for user in UserData.find(root):
        if user.name in config.users_skip: #
            # BDVGI: no end times
            # mtalpe: huge values
            # vhelm: duplicate times
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

        vals = []
        if not sheets: continue


    # rate_col = mass_mgr.as_collection().differentiate() + rate_mgr.as_collection().chunk_series()
    rate_col = rate_mgr.as_collection()
    process(mass_mgr.as_collection())
    # process(rate_col, config)

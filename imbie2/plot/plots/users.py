if __name__ == "__main__":
    import logging
    from basins import *
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    import sys
    from itertools import chain

    # logging.captureWarnings(True)
    logging.basicConfig(level=logging.WARNING)

    rate_mgr = MassRateCollectionsManager()
    mass_mgr = MassChangeCollectionsManager()

    if len(sys.argv) >= 2:
        root = sys.argv[1]
    else: root = None

    for user in UserData.find(root):
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

            items = [
                user.name, user.group, series.basin_group.value,
                series.basin_id.value, series.min_time, series.max_time,
                len(series)
            ]
            line = ','.join(str(i) for i in items)
            print(line)
            mass_mgr.add_series(series)
        #     if series is None: continue
        #     print('\t', series.min_time, series.max_time, series.basin_id.value, len(series))

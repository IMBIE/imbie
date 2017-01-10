#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def error_bars(dataset, ice_sheets, basins_group):
    styles = {
        'RA': {
            'width': .8,
            'mean': {
                'linewidth': 2,
                'color': 'white',
                'zorder': 3
            },
            'sigma1': {
                'color': '#ff0000',
                'zorder': 2
            },
            'sigma2': {
                'color': '#ff8888',
                'zorder': 1
            }
        },
        'GMB': {
            'width': .6,
            'mean': {
                'linewidth': 2,
                'color': 'white',
                'zorder': 6
            },
            'sigma1': {
                'color': '#00ff00',
                'zorder': 5
            },
            'sigma2': {
                'color': '#88ff88',
                'zorder': 4
            }
        }
    }

    fig = plt.gcf()
    axs = plt.gca()

    min_y = 0
    max_y = 0

    text_y = 20.5
    line_y = 20

    x = 0
    names = []

    for sheet in ice_sheets:
        n_basins = 0
        min_x = x

        for basin in basins_group.sheet(sheet):
            n_basins += 1
            names.append(basin.value)

            for group_data in dataset[basin]:
                if group_data.user_group not in styles:
                    continue
                style = styles[group_data.user_group]

                w = style['width']
                margin = 1. - (w / 2)

                # get min. and max. x
                x0 = x + margin

                # get values
                sigma = group_data.errs
                mean = group_data.dmdt

                # calc. pos and height for double err
                y0 = mean - (sigma * 2.)
                h = sigma * 4.

                # draw double err box
                r1 = mpatches.Rectangle(
                    (x0, y0), w, h, **style['sigma2']
                )
                axs.add_patch(r1)
                # draw single err box
                y0 = mean - sigma
                h = sigma * 2.

                r2 = mpatches.Rectangle(
                    (x0, y0), w, h, **style['sigma1']
                )
                axs.add_patch(r2)

                # draw mean line
                axs.plot(
                    [x0, x0+w], [mean, mean],
                    **style['mean']
                )

                min_y = min(min_y, mean - sigma*2)
                max_y = max(max_y, mean + sigma*2)

            x += 1

        if n_basins > 0:
            text_x = min_x + (n_basins / 2) + .5

            axs.text(
                text_x, text_y, sheet.value.upper(),
                horizontalalignment='center', zorder=7
            )
            axs.plot(
                [min_x+.6, x+.4], [line_y, line_y],
                'k-', linewidth=2, zorder=7
            )

    axs.axhline(0, color='black')
    axs.set_xlim(.5, x+.5)
    axs.set_ylim(min_y, max_y)

    axs.xaxis.tick_top()
    axs.set_xticks(np.arange(1, x+1))
    axs.set_xticklabels(names)

    plt.show()

if __name__ == "__main__":
    from basins import *
    from series import *
    from data_collections import *
    from random import Random

    r = Random()

    dset = MassRateCollectionsManager()
    for basin in ZwallyBasin:
        base_dM = r.random() * 30. - 20.

        dM = base_dM + r.random() * 5. - 2.5
        err = 4. + r.random() * 4.

        series = MassRateDataSeries(
            None, 'RA', None, BasinGroup.zwally,
            basin, None, None, None, None, dM, err
        )
        dset.add_series(series)

        dM = base_dM + r.random() * 5. - 2.5
        err = 6. + r.random() * 6.

        series = MassRateDataSeries(
            None, 'GMB', None, BasinGroup.zwally,
            basin, None, None, None, None, dM, err
        )
        dset.add_series(series)

    create_basins_plot(dset, [GenericBasin.apis, GenericBasin.eais, GenericBasin.wais], ZwallyBasin)

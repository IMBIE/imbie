from imbie2.plot.style import colours


def time_bars(ax, data, min_t=None, max_t=None):
        # min_t = 1990
        # max_t = 2020
    users = []
    starts = []
    lengths = []
    groups = []
    order = []
    i = 0
    for series in data:
        if series.computed: continue

        if min_t is None or min_t > series.min_time:
            min_t = series.min_time
        if max_t is None or max_t < series.max_time:
            max_t = series.max_time

        i += 1
        order.append(i / 5.)
        starts.append(series.min_time)
        lengths.append(series.max_time - series.min_time)
        users.append(series.user)
        groups.append(colours.primary[series.user_group])

    ax.barh(order, lengths, height=.1, left=starts, color=groups, tick_label=users)
    ax.set_xlim(min_t, max_t)
    ax.set_ylim(-.4, order[-1]+.4)
    # ax.minorticks_on()
    ax.xaxis.grid(b=True, which='major', linestyle='--')

    # create legend
    items = []
    labels = []
    for group in 'RA', 'GMB', 'IOM':
        name = grp_names[group]
        col = colours[group]

        items.append(
            patches.Patch(color=col, label=name)
        )
        labels.append(name)

    return items, labels

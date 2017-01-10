from matplotlib import patches
from imbie2.plot.style import colours


def coverage_boxes(ax, data, groups=None):
    min_t = 1990
    max_t = 2020
    min_r = None
    max_r = None
    if groups is None:
        groups = "RA", "GMB", "IOM"

    for series in data:
        if series.user_group not in groups:
            continue
        col = colours.secondary[series.user_group]

        r_len = series.max_rate - series.min_rate
        r_pos = series.min_rate
        t_len = series.max_time - series.min_time
        t_pos = series.min_time

        if r_len == 0:
            r_pos -= series.errs[0]
            r_len += series.errs[0]

        if min_t is None or min_t > t_pos:
            min_t = t_pos
        if max_t is None or max_t < t_pos + t_len:
            max_t = t_pos + t_len
        if min_r is None or min_r > r_pos:
            min_r = r_pos
        if max_r is None or max_r < r_pos + r_len:
            max_r = r_pos + r_len

        if series.computed:
            rect = patches.Rectangle(
                (t_pos, r_pos),
                t_len, r_len,
                edgecolor=col, hatch='\\/',
                fill=None
            )
        else:
            rect = patches.Rectangle(
                (t_pos, r_pos),
                t_len, r_len,
                facecolor=col, alpha=.4
            )
        ax.add_patch(rect)
    # create legend
    items = []
    labels = []

    for group in groups:
        col = colours.primary[group]

        items.append(
            patches.Patch(color=col, label=group, alpha=.5)
        )
        labels.append(group)

    r_range = abs(max_r - min_r)
    r_pad = .05 * r_range
    t_range = abs(max_t - min_t)
    t_pad = .01 * t_range

    ax.set_xlim([min_t-t_pad, max_t+t_pad])
    ax.set_ylim([min_r-r_pad, max_r+r_pad])

    ax.xaxis.tick_top()
    ax.axhline(0, color='black')

    return items, labels

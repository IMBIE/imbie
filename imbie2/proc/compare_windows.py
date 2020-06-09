from imbie2.model.collections import WorkingMassRateCollection, MassChangeCollection
from typing import Union, Dict, List
import numpy as np


Collection = Union[WorkingMassRateCollection, MassChangeCollection]


class WindowStats:
    def __init__(self, start: float, end: float, **groups: Dict[str, int]):
        self.start = start
        self.end = end
        self.groups = groups
        self.count = sum(n for n in groups.values())

    @property
    def length(self):
        return self.end - self.start

    @classmethod
    def count_series(cls, data: Collection, start: float, end: float) -> "WindowStats":
        groups = {}
        for series in data:
            if series.user_group not in groups:
                groups[series.user_group] = 0

            if series.min_time <= start and series.max_time >= end:
                groups[series.user_group] += 1
        return cls(start, end, **groups)


def compare_windows(data: Collection, limit: int=None) -> List[WindowStats]:
    """
    searches pairs of start/end time in data to find
    periods of best coverage

    :param data:
    :return:
    """

    min_year = int(data.min_time())
    max_year = int(data.max_time()) + 1
    items = []

    for y1 in range(min_year, max_year-1):
        for y2 in range(y1+1, max_year):
            window_stats = WindowStats.count_series(data, y1, y2)
            items.append(window_stats)

    items.sort(key=lambda s: s.count)
    selected = []
    current = None
    while items:
        item = items.pop(-1)

        if item.start == 2003 and item.end == 2012:
            # force inclusion of this period for
            # comparison w/ imbie 2012 coverage
            selected.append(item)
            continue

        if item.length < 5:
            continue
        if current is None:
            current = item
        elif current.count == item.count:
            if current.length < item.length:
                current = item
        else:
            selected.append(current)
            current = item
    items = selected

    if limit is not None:
        items = items[:limit]

    return items

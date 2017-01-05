#! /usr/bin/python3
import os
import json
import sys
from parser import MassChangeParser
from data_collections import MassChangeCollectionsManager

class AnswersData:
    @property
    def files(self):
        for qn_name in self._data["files"]:
            if self.filter is None or qn_name in self.filter:
                file_data = self._data["files"][qn_name]
                yield file_data["name"]

    @property
    def group(self):
        return self._data["group"]

    def __init__(self, filename, _filter=None):
        self.filename = filename
        self.filter = _filter
        self._data = None

    def open(self, filename=None):
        if filename is not None:
            self.filename = filename

        _file = open(self.filename, 'r')
        self._data = json.load(_file)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._data = None


def answers_search(root=None, _filter=None):
    if root is None:
        root = os.getcwd()

    for path, _, files in os.walk(root):
        if '.answers.json' in files:
            fpath = os.path.join(path, '.answers.json')
            with AnswersData(fpath, _filter) as f:
                for fname in f.files:
                    print(fname, f.group)
                    yield (os.path.join(path, fname), f.group)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import logging

    logging.captureWarnings(True)
    logging.basicConfig(level=logging.CRITICAL)

    def plot(basin, linestyle='b-', fill='blue', alpha=.5, width=1):
        t, m, e = basin.smooth()

        plt.fill_between(
            t, m-e, m+e,
            alpha=alpha, facecolor=fill,
            interpolate=True
        )
        plt.plot(t, m, linestyle, linewidth=width)

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = None
    mass = [
        # "mascons-approach-upload",
        # "mean-rate-upload"
        "spherical-harmonics-upload",
        "time-series-upload",
        "mass-balance-upload"
    ]

    mgr = MassChangeCollectionsManager()
    for ans, group in answers_search(folder, mass):

        with MassChangeParser(ans, group) as data:
            if data is None: continue

            for basin in data:
                mgr.add_series(basin)

    for col in mgr:
        for s in col:
            if s.user_group == 'GMB':
                l = '-g'
                f = 'green'
            else:
                l = '-b'
                f = 'blue'
            plot(s, l, f, alpha=.2)
        c = col.combine()
        plot(c, '-k', 'red', 1, 2)

        plt.grid()

        plt.title(c.basin_id.value)
        plt.xlabel("year")
        plt.ylabel("mass change (Gt)")

        plt.show()

#!/usr/bin/python3
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
import sys
import json
import csv


def plot_gia(series):
    _map = Basemap(
        projection='aeqd', lat_0=0, lon_0=0
    )
    _map.drawcoastlines()
    _map.drawmeridians(np.arange(0, 360, 30))
    _map.drawparallels(np.arange(-90, 90, 30))

    print(len(series))

    cols = ['r','g','b','c','m','y','k']

    for si, s in enumerate(series):
        c = cols[si % len(cols)]

        print(si, len(series))

        n = 0
        for lat, lon, vel in zip(s.lat, s.lon, s.vel):
            if abs(vel) < .01 or lat > -80:
                continue

            n += 1
            x, y = _map(lon, lat)
            _map.plot(x, y, ','+c)

        print(n)
    plt.show()

def read_user(path):
    fname = os.path.join(path, '.answers.json')
    data_file = None

    with open(fname) as f:
        data = json.load(f)

        try:
            data_file = data['files']['uplift-rates-data']['name']
        except KeyError: return None
    if data_file is None:
        return None

    dpath = os.path.join(path, data_file)

    try:
        return pd.read_csv(
            dpath, header=None, names=['lon','lat','vel'],
            delim_whitespace=True, usecols=[0,1,2]
        )
    except: return None

if __name__ == "__main__":
    root = sys.argv[1]
    data = []
    for path, _, files in os.walk(root):
        if '.answers.json' in files:
            print(path, '...', end='\r')
            dset = read_user(path)
            if dset is not None:
                data.append(dset)
                print(path, '/')
            else: print(path, 'X')

    plot_gia(data)

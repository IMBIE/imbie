from .base import DataLoader

import numpy as np
from collections import namedtuple
import os
import re


RacmoData = namedtuple("RacmoData", ["time", "smb_tot", "smb_regions"])

class RacmoLoader(DataLoader):
    """
    Loads the Racmo data
    """
    folder = os.path.join("racmo", "AIS_basins_ASCII")
    fpattern = "zwally"

    def __init__(self, root=None, folder=None, fpattern=None):
        """
        Initializes the RacmoLoader

        INPUTS:
            root: (optional) The root directory
            folder: (optional) The sub-folder in which to search
            fpattern: (optional) The string to match filenames against
        """
        # init. base class
        DataLoader.__init__(self, root)
        # override folder
        if folder is not None:
            self.folder = folder
        # override default pattern
        if fpattern is not None:
            self.fpattern = fpattern

    def read(self):
        """
        Reads the Racmo files
        """
        # create full folder path
        path = os.path.join(self.root, self.folder)

        # list for storing matched filename
        fnames = []
        # list for storing id numbers of matched filenames
        ids = []
        # create regex for finding id numbers in filenames
        #   e.g: matches the '12' in 'abcde_fghi.jkl_12.txt',
        #        or '3' in 'another.file.name.3.txt'
        id_finder = re.compile("[0-9]+(?=.txt)")

        # get each file name in the folder
        for fname in os.listdir(path):
            # if it matches the string to search for...
            if self.fpattern in fname:
                # ...add it to the list of file names,
                fnames.append(fname)
                # and find its ID number,
                id_match = re.search(id_finder, fname)
                # then add that ID number to the list
                ids.append(id_match.group())
        # convert the file names list to a numpy array
        #   (so that we can use fancy indexing)
        fnames = np.asarray(fnames)

        # convert the ids list to a numpy array, and cast its
        #   values into integers
        ids = np.asarray(ids, dtype=int)
        # get the indices sorted by id number (ascending)
        order = np.argsort(ids)

        # sort the file names into the same order
        fnames = fnames[order]
        # get the number of file names
        n_basins, = fnames.shape

        # create a function for converting the date field into the
        #   correct format (YYYYMMDD into YYYY.YY)
        def date_convert(t):
            t = str(t)
            return float(t[0:4]) +\
                   float(t[4:6]) / 12. +\
                   float(t[6:8]) / 365.25

        # load all the contents of the first file
        data = self.read_csv(os.path.join(path, fnames[0]), delim='\t')
        # get the date column
        date = data[0, :]
        # convert the values and store as a numpy array
        time = np.asarray(
            [date_convert(t) for t in date], dtype='float64'
        )
        # get the number of items
        n_time, = time.shape
        # create empty smb array
        smb = np.zeros([n_time, n_basins])

        for i in range(n_basins):
            # get file path of each file
            fpath = os.path.join(path, fnames[i])
            # load the contents of the file
            data = self.read_csv(fpath, delim='\t')
            # and store its values as a column in the smb array
            smb[:, i] = data[1, :] * 12.

        # digitise MAR from lenearts et al 2012 fig 3
        ais_tot = np.array(
            [2325, 2425, 2463, 2463, 2400, 2625, 2550, 2388, 2380, 2537, 2537, 2280, 2362, 2712, 2460, 2300, 2312,
             2410, 2240, 2537, 2375, 2360, 2575, 2360, 2337, 2500, 2525, 2400, 2175, 2437, 2450, 2300])
        time_tot = np.arange(32) + 1979
        ok = np.flatnonzero(time_tot > 1990)
        R = np.mean(ais_tot) / np.mean(ais_tot[ok])

        # compute MAR & anomaly
        msmb = np.empty([n_basins])
        dsmb = np.empty([n_time, n_basins])
        for i in range(n_basins):
            msmb[i] = np.mean(smb[:, i]) * R
            dsmb[:, i] = smb[:, i] - msmb[i]

        # aggregate into regions
        regions = ['EAIS', 'WAIS', 'APIS']
        n_regions = 3

        smb_regions = np.empty([n_time, n_regions])
        dsmb_regions = np.empty([n_time, n_regions])
        msmb_regions = np.empty([n_regions])
        ncells_regions = np.empty([n_regions])

        # FIXME: missing variables region_names_z12 & regions_z12 - need to investigate this
        # region_names_z12 = []
        # regions_z12 = []
        # for i, region in enumerate(regions):
        #     ok1 = np.flatnonzero(region_names_z12 == region)
        #     ok2 = np.flatnonzero(regions_z12 == i+1)
        #     ncells_regions[i] = len(ok2)
        #
        #     for j in ok1:
        #         smb_regions[:, i] = smb_regions[:, i] + smb[:, j]
        #         dsmb_regions[:, i] = dsmb_regions[:, i] + dsmb[:, j]
        #     msmb_regions[i] = np.mean(smb_regions[:, i])

        # aggregate to AIS
        smb_tot = np.tile(smb_regions, (3, 1))
        # variables below are NOT USED
        # dsmb_tot = np.tile(dsmb_regions, (3, 1))
        # msmb_tot = np.sum(msmb_regions)

        # store values in tuple and return it
        self.data = RacmoData(time, smb_tot, smb_regions)
        return self.data
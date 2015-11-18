from .base import DataLoader

import numpy as np
import os
from collections import namedtuple


RAData = namedtuple("RAData", ["EAIS", "WAIS"])
RAAVSIS = namedtuple("RAAVSIS", ["dMstar", "dMstar_sd", "dMstar_firn", "dMstar_firn_sd"])
RAMMIS = namedtuple("RAMMIS", ["time", "dM", "dM_sd"])

__author__ = 'Mark'

NTOT_EAIS = 99111L
NTOT_WAIS = 17490L
NINTERP_EAIS = 73416L
NINTERP_WAIS = 13266L

class RALoader:
    """
    Loads both the AVS and the MM radar altimetry data
    """
    def __init__(self, root=None, mm_folder=None, mm_fname=None, avs_folder=None, **avs_fnames):
        """
        Initialize the RALoader instance by creating an AVS loader and an RA loader

        INPUTS:
            root: (optional) override root folder
            mm_folder: (optional) override MM sub-folder
            mm_fname: (optional) override the MM filename
            avs_folder: (optional) override the AVS sub-folder
            **avs_fnames: (optional) override the AVS filenames
        """
        self.avs_loader = RAAVSLoader(root, avs_folder, **avs_fnames)
        self.ra_loader = RAMMLoader(root, mm_folder, mm_fname)

    def read(self):
        """
        Load the AVS data then use it to load the MM data and return that
        result.
        """
        self.avs = self.avs_loader.read()
        self.data = self.ra_loader.read(self.avs)
        return self.data

class RAAVSLoader(DataLoader):
    """
    Loads the RA AVS data
    """
    rho_ice = .917
    rho_snow = .4

    # default file names
    fnames = {
        "EAIS": "antarctic_alt_results EAIS.txt",
        "WAIS": "antarctic_alt_results WAIS.txt"
    }
    # default sub-folder
    folder = os.path.join("altimetry", "data", "avs_final")

    def __init__(self, root=None, folder=None, **fnames):
        """
        Initialize the RAAVSLoader

        INPUTS:
            root: (optional) override root folder
            folder: (optional) override the AVS sub-folder
            **fnames: (optional) override the AVS filenames
        """
        # call default init.
        DataLoader.__init__(self, root)
        # override the sub-folder
        if folder is not None:
            self.folder = folder
        # override any filenames provided
        for k, fname in fnames.items():
            if k in self.fnames.keys():
                self.fnames[k] = fname

    def read_file(self, fpath, ntot, ninterp, rho):
        """
        reads an AVS file and returns the data

        INPUTS:
            fpath: the full path of the file to be read
            ntot: either NTOT_EAIS or NTOT_WIAS
            ninterp: either NINTERP_EAIS or NINTERP_WAIS
            rho: either self.rho_ice or self.rho_snow
        """
        # read contents of file
        data = self.read_csv(fpath, header=1)

        # select columns
        dh = data[1, :]
        dh_sd = data[2, :]
        dh_firn = data[3, :]

        dfirn = dh - dh_firn
        dfirn_sd = np.abs(dfirn) * .15
        dM = (dh / 1e5) * 100 * ninterp * rho
        dM_sd = (dh_sd / 1e5) * 100 * ninterp * rho
        # FIXME: confirm if 'rho_ice' below is the correct behaviour
        dM_firn = (dh_firn / 1e5) * 100 * ninterp * self.rho_ice
        dM_firn_sd = ( np.sqrt(dfirn_sd ** 2. + dh_sd ** 2.) / 1e5 ) * 100 * ninterp * self.rho_ice

        dMstar = dM * ntot / ninterp
        dMstar_sd = dM_sd * ntot / ninterp
        dMstar_firn = dM_firn * ntot / ninterp
        dMstar_firn_sd = dM_firn_sd * ntot / ninterp

        # create & return tuple
        return RAAVSIS(dMstar, dMstar_sd, dMstar_firn, dMstar_firn_sd)

    def read(self):
        """
        read both AVS files and return the outputs
        """
        # get path to EAIS file
        eais_path = os.path.join(self.root, self.folder, self.fnames["EAIS"])
        # get path to WAIS file
        wais_path = os.path.join(self.root, self.folder, self.fnames["WAIS"])

        # read files and store results in a tuple
        self.data = RAData(
            EAIS=self.read_file(eais_path, NTOT_EAIS, NINTERP_EAIS, self.rho_snow),
            WAIS=self.read_file(wais_path, NTOT_WAIS, NINTERP_EAIS, self.rho_ice)
        )
        return self.data


class RAMMLoader(DataLoader):
    """
    loads the MM RA data
    """
    # default filename
    fname = "mass_timeseries.Ers1Ers2Env.reg_z11.txt"
    # default sub-folder
    folder = os.path.join("altimetry", "data", "mm_final")
    # override defulat CS delimiter
    delim = ' '

    def __init__(self, root=None, folder=None, fname=None):
        """
        initialize the RA MM Loader
        """
        # init. base class, overriding root if provided
        DataLoader.__init__(self, root)
        # override default sub-folder
        if folder is not None:
            self.folder = folder
        # override default filename
        if fname is not None:
            self.fname = fname

    def read(self, ra_avs_data):
        """
        read the MM data, and over-write some values using data
        from an RAAVSLoader instance.
        """
        fpath = os.path.join(self.root, self.folder, self.fname)
        data = self.read_csv(fpath, header=1)

        _id = data[0, :]
        time = data[1, :]
        dM = data[2, :] * NTOT_WAIS / NINTERP_WAIS
        # dM_sd = data[3, :] * NTOT_WAIS / NINTERP_WAIS ( not used )

        # find indices for EAIS
        eais = np.flatnonzero(_id == 1)
        # find time & dm for EAIS
        time_eais = time[eais]
        dM_eais = dM[eais]
        # get std. dev. from AVS
        dM_sd_eais = ra_avs_data.EAIS.dMstar_sd

        # find indicies for WAIS
        wais = np.flatnonzero(_id == 2)
        # find time & dm for WAIS
        time_wais = time[wais]
        dM_wais = dM[wais]
        # get std. dev. from AVS
        dM_sd_wais = ra_avs_data.WAIS.dMstar_sd

        # create & return tuples
        self.data = RAData(
            EAIS=RAMMIS(time=time_eais, dM=dM_eais, dM_sd=dM_sd_eais),
            WAIS=RAMMIS(time=time_wais, dM=dM_wais, dM_sd=dM_sd_wais)
        )
        return self.data
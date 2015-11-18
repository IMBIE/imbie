from .base import DataLoader

from collections import namedtuple
import os


# tuple for storing Ice Sheet data
GraceIS = namedtuple("GraceIS", ["month", "mass", "date", "ngroups"])
# tuple for storing overall GRACE data
GraceData = namedtuple("GraceData", ["APIS", "EAIS", "WAIS", "GrIS", "AIS"])


class GraceLoader(DataLoader):
    """
    Class for loading GRACE data
    """
    # set default subfolder path
    folder = os.path.join("GRACE", "grace_final_numbers.dir")
    # set default file names for each ice sheet
    fnames = {
        "APIS": "out.average.Antarctic Peninsula",
        "EAIS": "out.average.East Antarctica",
        "WAIS": "out.average.West Antarctica",
        "GrIS": "out.average.Greenland",
        "AIS": "out.average.Antarctica"
    }
    # override default CSV delimiter
    delim = ' '

    def __init__(self, root=None, folder=None, **fnames):
        """
        Initializes the GraceLoader instance

        INPUTS:
            root: (optional) override default root directory
            folder: (optional) override default sub-folder
            **fnames: (optional) override default file names
        """
        # call base class init.
        DataLoader.__init__(self, root)

        # override folder value
        if folder is not None:
            self.folder = os.path.expanduser(folder)
        # override fnames.
        for k, fname in fnames.items():
            if k in self.fnames.keys():
                self.fnames[k] = fname

    def read_file(self, fpath):
        """
        reads a GRACE file.

        INPUTS:
            fpath: the full path to the input file
        """
        # open the file
        with open(fpath, 'r') as f:
            # read first line to get number of elements
            count = int(f.readline())
        # use the read_csv method to retreive the data
        data = self.read_csv(fpath, header=1, limit=count)

        # get columns
        month = data[0, :]
        mass = data[1, :]
        date = data[2, :]
        ngroups = data[3, :]
        # return data for this ice sheet
        return GraceIS(month, mass, date, ngroups)

    def read(self):
        """
        Reads all the input files
        """
        data = {}
        for isname, fname in self.fnames.items():
            # for each ice sheet, find the full file path
            fpath = os.path.join(self.root, self.folder, fname)
            # load the contents of the file
            data[isname] = self.read_file(fpath)
        # create and return tuple of the data.
        self.data = GraceData(**data)
        return self.data
from itertools import islice
import numpy as np
import csv
import os


class DataLoader:
    """
    This is the base class from which the various data loaders inherit.
    This class doesn't have complete functionality and shouldn't be
    used directly.
    """
    # set default root directory
    root = os.path.expanduser("~\\Downloads\\Shepherd IMBIE\\experiments")
    # set default CSV delimiter
    delim = '\t'

    def __init__(self, root=None):
        """
        initializes the DataLoader

        INPUTS:
            root: (optional) overrides the default root directory
        """
        if root is not None:
            self.root = root
        self.data = None

    def read_csv(self, fname, header=0, limit=None, delim=None, max_width=None, cast_float=True):
        """
        function for reading CSV files

        INPUTS:
            fname: the full path to the CSV file
            header: (optional) the number of header lines to skip
            limit: (optional) the maximum number of lines to read (incl. header lines)
            delim: (optional) override the default CSV delimiter
            max_width: (optional) the maximum number of data items per row
            cast_float: (optional) if True, returns an array of floats instead of strings.
        """
        # get default delimiter
        if delim is None:
            delim = self.delim

        # create function for removing whitespace & empty items from a line of data
        def clean_items(row):
            row = [item.strip() for item in row if item]
            if max_width is not None:
                return row[:max_width]
            return row

        # open the file
        with open(fname, 'rb') as f:
            # create a CSV reader instance
            reader = csv.reader(f, delimiter=delim)
            # read the file
            data = [clean_items(line) for line in islice(reader, header, limit)]
        # return the file contents as a numpy array
        if cast_float:
            return np.asarray(data, dtype='float64').T
        return np.asarray(data)

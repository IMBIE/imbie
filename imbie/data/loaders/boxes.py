from .base import DataLoader

import os
import numpy as np

class BoxesLoader(DataLoader):
    """
    Loads the data for constructing the box-plot chart
    from a specified CSV file
    """

    folder = '.'
    fname = 'boxes.csv'

    def __init__(self, root=None, folder=None, fname=None):
        """
        Initialize
        """
        # call base class init.
        DataLoader.__init__(self, root)
        # override folder & file names
        if folder is not None:
            self.folder = folder
        if fname is not None:
            self.fname = fname

    def read(self):
        """
        Loads the data from the CSV file
        """
        # create full path
        path = os.path.join(self.root, self.folder, self.fname)
        # load CSV data. DO NOT try to cast to floats, as we need
        #   some of the string information
        data = self.read_csv(path, delim=',', cast_float=False)

        # init. output dicts.
        dmdt = {}
        dmdt_sd = {}

        # get the names of the sheets and the data sources
        sheets = [s.strip() for s in data[1:, 0]]
        sources = [s.strip() for s in data[0, 1:]]
        # this will store the output order of the sources
        unique_sources = []

        for i, sheet in enumerate(sheets):
            # create empty dicts for this sheet
            dmdt[sheet] = {}
            dmdt_sd[sheet] = {}

            for j, source in enumerate(sources):
                # get the numeric value for this source & sheet
                value = data[i+1, j+1].strip()
                # if no value, store a NaN.
                if not value:
                    value = np.NaN
                else:
                    value = float(value)
                # check if source is a std. dev. value
                #  insert value to appropriate dictionary
                if source[-3:] == '_sd':
                    source = source[:-3]
                    dmdt_sd[sheet][source] = value
                else:
                    dmdt[sheet][source] = value
                # add the source to the list, if we haven't already
                if source not in unique_sources and source != "All":
                    unique_sources.append(source)
        # return the data
        return dmdt, dmdt_sd, sheets, list(unique_sources)
from .base import DataLoader

from collections import namedtuple
import os
from xml.etree import ElementTree as et

__author__ = 'Mark'

DMData = namedtuple("DMData", ["EAIS", "WAIS", "APIS"])

class DMLoader(DataLoader):
    """
    Loads the DM data. In the original IMBIE 1 IDL code, all these
    values were hard-coded. This has now been replaced with a method
    that loads the values from an XML file.

    The original behaviour remains in the .read_static_data method, but
    the XML-loading .read method is preferred.
    """
    fname = 'dm.xml'

    def __init__(self, root=None, fname=None):
        """
        Initialize the DMLoader instance
        """
        # call default init.
        DataLoader.__init__(self, root)
        if fname is not None:
            self.fname = fname

    def read(self):
        """
        Reads the DM data from an XML file and
        returns a tuple of the results
        """
        # create full path
        fpath = os.path.join(self.root, self.fname)

        # create tree parser
        tree = et.parse(fpath)
        root = tree.getroot()

        # init. dict that data will be stored in
        data = {}

        for sheet_elem in root:
            # step through each element in the file
            # if it's not a known ice sheet, skip it.
            if sheet_elem.tag not in ['EAIS', 'WAIS', 'APIS']:
                continue
            # get the values of the sub-elems
            ntot = float(sheet_elem.find('ntot').text)
            msmb = float(sheet_elem.find('msmb2_ng').text)

            # calculate the dsmb_sd
            n = ntot * 10 * 10 / (200 ** 2.)
            dsmb_sd = (msmb * .15 / n ** .5) * 2 ** .5
            # add it to the dict
            data[sheet_elem.tag] = dsmb_sd
        # create & return the tuple
        self.data = DMData(**data)
        return self.data

    def read_static_data(self):
        """
        return the DM data.
        """
        ntot_APIS = 2289.
        ntot_EAIS = 99111.
        ntot_WAIS = 17490.
        # msmb_regions_ng = [962.208, 459.294, 123.956] NOT USED
        msmb2_regions_ng = [615.915, 267.836, 153.711]
        n_EAIS = ntot_EAIS * 10 * 10 / (200 ** 2.)
        n_WAIS = ntot_WAIS * 10 * 10 / (200 ** 2.)
        n_APIS = ntot_APIS * 10 * 10 / (200 ** 2.)
        dsmb_sd_EAIS_ng = (msmb2_regions_ng[0] * 0.15 / n_EAIS ** 0.5) * 2 ** 0.5
        dsmb_sd_WAIS_ng = (msmb2_regions_ng[1] * 0.15 / n_WAIS ** 0.5) * 2 ** 0.5
        dsmb_sd_APIS_ng = (msmb2_regions_ng[2] * 0.15 / n_APIS ** 0.5) * 2 ** 0.5

        self.data = DMData(EAIS=dsmb_sd_EAIS_ng, WAIS=dsmb_sd_WAIS_ng, APIS=dsmb_sd_APIS_ng)
        return self.data
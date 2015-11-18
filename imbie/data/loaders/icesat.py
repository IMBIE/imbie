from .base import DataLoader
from imbie.functions import ts2m

import numpy as np
from collections import namedtuple
import os
from xml.etree import ElementTree as et


ICESatIS = namedtuple("ICESatIS", ["t", "dmdt", "dmdt_sd"])
ICESatData = namedtuple("ICESatData", ["EAIS", "WAIS", "APIS", "GrIS", "dmdt_sd", "dmdt_sd_GrIS_raw"])

class ICESatLoader(DataLoader):
    """
    Loads the ICESat data. In the original IMBIE 1 IDL code, all these
    values were hard-coded, but these values have been moved into an XML
    file, and this class now loads the data from that file.

    The original behaviour remains in the .read_static_data method, but
    the XML-loading .read method is preferred.
    """
    fname = "icesat.xml"

    def __init__(self, root, fname=None):
        """
        Initialize the ICESatLoader instance
        """
        # call default init.
        DataLoader.__init__(self, root)
        if fname is not None:
            self.fname = fname

    def read(self):
        """
        Reads the ICESat data from an XML file, using the
        xml.etree.ElementTree XML parser.
        """
        # create full path
        fpath = os.path.join(self.root, self.fname)
        # create the tree parser
        tree = et.parse(fpath)
        root = tree.getroot()

        # create convenience function for converting text strings
        #   to numpy arrays
        def nparray(text):
            return np.asarray(
                text.split(','), dtype='float64'
            )

        # find the time data
        t_elem = tree.find('t')
        # convert it to an array
        t_icesat = nparray(t_elem.text)

        # prepare the dictionary that will be used to initialize
        #   the output tuple
        data = {
            'dmdt_sd': []
        }
        # step through each element in the file
        for elem in root:
            if elem.tag not in ['EAIS', 'WAIS', 'APIS', 'GrIS']:
                # skip to next elem, it isn't a known ice sheet
                continue
            if 'raw' in elem.attrib and elem.attrib['raw'] == 'True':
                # process 'raw' data - as GrIS data in method below.
                # load the time values
                t0 = nparray(elem.find('t0').text)
                t1 = nparray(elem.find('t1').text)
                # find the average
                t = (t0 + t1) / 2.

                # load the dmdt values
                dmdt = nparray(elem.find('dmdt').text)
                # interpolate to monthly
                t, dmdt = ts2m(t, dmdt)

                key = "dmdt_sd_{}_raw".format(elem.tag)
                # load the dmdt_sd data, and add it to the output
                #   dict
                data[key] = nparray(elem.find('dmdt_sd').text)

                dmdt_sd = t * 0. + data[key][0]
                # create tuple for this ice sheet & add to dict
                data[elem.tag] = ICESatIS(t, dmdt, dmdt_sd)
            else:
                # for normal (non-raw) behaviour
                # find dmdt & sd values
                dmdt = float(elem.find('dmdt').text)
                dmdt_sd = float(elem.find('dmdt_sd').text)

                # add the dmdt value to the dmdt array
                data['dmdt_sd'].append(dmdt)

                # interpolate to monthly
                t, dmdt = ts2m(t_icesat, np.repeat([dmdt], 2))
                dmdt_sd = t * 0. + dmdt_sd
                # create output tuple & add to dict
                data[elem.tag] = ICESatIS(t, dmdt, dmdt_sd)

        # convert the SD list to a numpy array
        data['dmdt_sd'] = np.asarray(data['dmdt_sd'], dtype='float64')
        # create & return output tuple
        self.data = ICESatData(**data)
        return self.data

    def read_static_data(self):
        """
        returns the ICESat values. This function has now been replaced
        by the .read method above, which loads the same values from an
        XML file.
        """
        t_icesat = np.array([2003.75, 2008.75])
        dmdt_icesat = np.array([109, -60, -28])
        data = {
            "dmdt_sd": np.array([57, 39, 18]),
            "dmdt_sd_GrIS_raw": np.sqrt((np.array([23, 23, 23, 23]) ** 2.) + 8. ** 2.)
        }
        t0_icesat_GrIS = np.array([2003, 2004, 2005, 2006])
        t1_icesat_GrIS = np.array([2006, 2007, 2008, 2009])
        t_icesat_GrIS_raw = (t0_icesat_GrIS + t1_icesat_GrIS) / 2.
        dmdt_icesat_GrIS_raw = (np.array([-178, -184, -216, -210]) - 174) / 2.
        # dmdt_icesat_GrIS_gic_raw = [-217, -253, -282, -216] NOT USED

        # interpolate icesat data to monthly
        t_EAIS_icesat, dmdt_EAIS_icesat = ts2m(t_icesat, np.repeat([dmdt_icesat[0]], 2))
        dmdt_sd_EAIS_icesat = t_EAIS_icesat * 0. + data["dmdt_sd"][0]
        data["EAIS"] = ICESatIS(t_EAIS_icesat, dmdt_EAIS_icesat, dmdt_sd_EAIS_icesat)

        t_WAIS_icesat, dmdt_WAIS_icesat = ts2m(t_icesat, np.repeat([dmdt_icesat[1]], 2))
        dmdt_sd_WAIS_icesat = t_WAIS_icesat * 0. + data["dmdt_sd"][1]
        data["WAIS"] = ICESatIS(t_WAIS_icesat, dmdt_WAIS_icesat, dmdt_sd_WAIS_icesat)

        t_APIS_icesat, dmdt_APIS_icesat = ts2m(t_icesat, np.repeat([dmdt_icesat[2]], 2))
        dmdt_sd_APIS_icesat = t_APIS_icesat * 0. + data["dmdt_sd"][2]
        data["APIS"] = ICESatIS(t_APIS_icesat, dmdt_APIS_icesat, dmdt_sd_APIS_icesat)

        t_GrIS_icesat, dmdt_GrIS_icesat = ts2m(t_icesat_GrIS_raw, dmdt_icesat_GrIS_raw)
        dmdt_sd_GrIS_icesat = t_GrIS_icesat * 0. + data["dmdt_sd_GrIS_raw"][0]
        data["GrIS"] = ICESatIS(t_GrIS_icesat, dmdt_GrIS_icesat, dmdt_sd_GrIS_icesat)

        self.data = ICESatData(**data)
        return self.data
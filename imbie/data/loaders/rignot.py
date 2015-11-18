from .base import DataLoader

from collections import namedtuple
import os

RignotIS = namedtuple("RignotIS",
                      ["SMB", "SMB_s", "D", "D_s", "TMB", "TMB_s", "Cumul_IOM", "Cumul_IOM_s", "dmdt_sd_iom"])
RignotData = namedtuple("RignotData",
                        ["GrIS", "WAIS", "EAIS", "APIS", "TMB", "TMB_s", "DATE_IOM", "DATE_R11", "DATE_R12",
                         "TMBGrISGRL2011", "TMBGrISGRL2011_s", "CumulAIS_IOM", "CumulAIS2012_IOM"])


class RignotLoader(DataLoader):
    """
    Loads the RIGNOT data
    """
    folder = "rignot"
    fname = "IMBIE_IOM_Final_2012.csv"

    delim = ","

    # these nested dicts are used to identify which column contains
    #   the data for each value of each ice sheet
    columns = {
        "GrIS": {
            "SMB": 2,
            "SMB_s": 3,
            "D": 4,
            "D_s": 5,
            "TMB": 6,
            "TMB_s": 7,
            "Cumul_IOM": 42,
            "Cumul_IOM_s": 43,
            "dmdt_sd_iom": 7},
        "WAIS": {
            "SMB": 9,
            "SMB_s": 10,
            "D": 15,
            "D_s": 16,
            "TMB": 21,
            "TMB_s": 22,
            "Cumul_IOM": 29,
            "Cumul_IOM_s": 30,
            "dmdt_sd_iom": 22
        },
        "EAIS": {
            "SMB": 11,
            "SMB_s": 12,
            "D": 17,
            "D_s": 18,
            "TMB": 23,
            "TMB_s": 24,
            "Cumul_IOM": 33,
            "Cumul_IOM_s": 34,
            "dmdt_sd_iom": 24},
        "APIS": {
            "SMB": 13,
            "SMB_s": 14,
            "D": 19,
            "D_s": 20,
            "TMB": 25,
            "TMB_s": 26,
            "Cumul_IOM": 37,
            "Cumul_IOM_s": 38,
            "dmdt_sd_iom": 26
        }
    }

    def __init__(self, root=None, folder=None, fname=None):
        """
        Initializes the RignotLoader

        INPUTS:
            root: (optional) the root directory
            folder: (optional) the sub-folder
            fname: (optional) the file name
        """
        # init. base class
        DataLoader.__init__(self, root)
        # override folder value
        if folder is not None:
            self.folder = folder
        # override file name
        if fname is not None:
            self.fname = fname

    def read(self):
        """
        Reads the contents of the rignot file
        """
        # create full file path
        fpath = os.path.join(self.root, self.folder, self.fname)
        # read the file
        data = self.read_csv(fpath, header=2, limit=221, max_width=46)

        # ROOT VALUES
        # these parameters don't have individual values for each
        #   ice sheet, so they'll be stored at the top level of
        #   the output tuple
        values = {
            "DATE_IOM": data[1, :],
            "DATE_R11": data[1, :],
            "DATE_R12": data[1, :],
            "TMB": data[27, :],
            "TMB_s": data[28, :],
            "CumulAIS_IOM": data[41, :],
            "CumulAIS2012_IOM": data[41, :]
        }
        # ICE SHEET VALUES
        # these parameters have one value for each ice sheet
        for is_name in self.columns:
            # get each ice sheet identifier,
            # and the column indices for that sheet
            column_set = self.columns[is_name]
            # the dict constructor below creates key/value pairs by iterating
            #   through the selected sub set of self.columns (defined above).
            # Each key in the column_set is the name of the property, and the value
            #   is the column index for that value. This constructor retrieves that
            #   column, and adds it to the dictionary under the same key.
            is_values = {k: data[col, :] for k, col in column_set.items()}
            # use the constructed dict to create a tuple for the current ice sheet
            values[is_name] = RignotIS(**is_values)
        # add the final two params to the values dictionary
        #   (this has to be done after the sheets have been loaded)
        values.update({
            "TMBGrISGRL2011": values["GrIS"].TMB,
            "TMBGrISGRL2011_s": values["GrIS"].TMB_s,
        })
        # create & return a tuple from the values dict
        self.data = RignotData(**values)
        return self.data

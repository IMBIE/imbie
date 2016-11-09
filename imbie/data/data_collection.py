from .source_collections import *
from .loaders import *

class DataCollection:
    """
    This class creates Loader instances for each of the
    input sources, and collates the data. Instances of this
    class will be used the imbie.processing.Processor to retrieve
    the input data.
    """
    grace = None
    icesat = None
    ra = None
    racmo = None
    rignot = None
    dm = None

    def __init__(self, **config):
        """
        Initializes the DataCollection, creating Loader
        instances for each input source.

        INPUTS:
            root: (optional) the root directory from which to
                    load data files.
        """
        # create *Loader instances
        self.grace_loader = GraceLoader(
            **config.get('grace', {})
        )
        self.icesat_loader = ICESatLoader(
            **config.get('icesat', {})
        )
        self.ra_loader = RALoader(
            **config.get('ra', {})
        )
        self.racmo_loader = RacmoLoader(
            **config.get('racmo', {})
        )
        self.rignot_loader = RignotLoader(
            **config.get('rignot', {})
        )
        self.dm_loader = DMLoader(
            **config.get('dm', {})
        )
        self.boxes_loader = BoxesLoader(
            **config.get('boxes', {})
        )

    def read(self):
        """
        instructs each loader to load the data from its input
        files, and stores the outputs.
        """
        self.grace = self.grace_loader.read()
        self.icesat = self.icesat_loader.read()
        self.ra = self.ra_loader.read()
        self.racmo = self.racmo_loader.read()
        self.rignot = self.rignot_loader.read()
        self.dm = self.dm_loader.read()
        self.boxes = self.boxes_loader.read()

    def get_grace(self, sheet_id, method):
        """
        get the GRACE data for the requested ice sheet

        INPUTS:
            sheet_id: the abbreviated name of the ice sheet
            method: FIXED / VARIABLE - the method argument passed
                    to the GRACEData initializer.
        """
        data = getattr(self.grace, sheet_id)
        return GRACEData(sheet_id, data, method)

    def get_icesat(self, sheet_id):
        """
        get the ICESat data for the requested ice sheet

        INPUTS:
            sheet_id: the abbreviated name of the ice sheet
        """
        data = getattr(self.icesat, sheet_id)
        return ICESatData(data)

    def get_iom(self, sheet_id):
        """
        get the input/output data for the requested ice sheet

        INPUTS:
            sheet_id: the abbreviated name of the ice sheet
        """
        iom_icesheet = getattr(self.rignot, sheet_id)
        return IOMData(
            self.rignot.DATE_IOM,
            iom_icesheet.TMB,
            iom_icesheet.TMB_s,
            iom_icesheet.Cumul_IOM
        )

    def try_get_ra(self, sheet_id):
        """
        Get the radar altimetry data for the requested ice sheet
        (if it exists)

        INPUTS:
            sheet_id: the abbreviated name of the ice sheet
        """
        if hasattr(self.ra, sheet_id):
            ra_data = getattr(self.ra, sheet_id)
            dm_data = getattr(self.dm, sheet_id)
            return RAData(ra_data, dm_data)
        return None

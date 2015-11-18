import numpy as np

from ..functions import deriv_imbie, smooth_imbie, move_av
from ..consts import *


class SourceCollection:
    """
    Base class from which other source collection classes
    inherit.

    These classes are designed to provide aliases for the
    loaded data, as well as in some cases performing some
    minimal processing on that data.

    This is roughly equivalent to the 'ASSIGN' steps in the
    IMBIE 1 IDL code.

    These classes ensure that for each data source, values
    exist for:
        t (time)
        dmdt (dm/dt - rate of change of mass)
        dmdt_sd (standard deviation of dm/dt)
        cumul (cumulative change in mass)
    """
    def __init__(self, t, dmdt, dmdt_sd, cumul):
        """
        The base class initializer - assigns values
        for each of the parameters

        INPUTS:
            t: time abscissas
            dmdt: dm/dt values
            dmdt_sd: std. dev. of dm/dt
            cumul: cumulative mass change
        """
        self.t = t
        self.dmdt = dmdt
        self.dmdt_sd = dmdt_sd
        self.cumul = cumul

    def as_dict(self):
        return {
            "t": list(self.t),
            "dmdt": list(self.dmdt),
            "dmdt_sd": list(self.dmdt_sd),
            "cumul": list(self.cumul)
        }


class RAData(SourceCollection):
    """
    Source collection for Radar Altimetry data.
    loads from the RA MM, RA AVS, and DM data.
    """
    def __init__(self, ra_data, dm_data):
        """
        Calculates the dmdt, std. dev. and cumul. values.

        INPUTS:
            ra_data: the radar altimetry data tuple from an
                     RALoader instance
            dm_data: the data from a DMLoader instance
        """
        t = ra_data.time
        # calc. the derivative of the mass values to find dm/dt
        dmdt = deriv_imbie(t, ra_data.dM, width=WIDTH, clip=CLIP)
        # calc. the standard deviation
        dmdt_sd = np.sqrt(ra_data.dM_sd ** 2. + dm_data ** 2.)
        # perform moving average to smooth mass values
        cumul = smooth_imbie(t, ra_data.dM, width=WIDTH)

        # call the base class initializer
        SourceCollection.__init__(self, t, dmdt, dmdt_sd, cumul)


class GRACEData(SourceCollection):
    """
    The source collection class for GRACE (gravimetry) data.
    """
    def __init__(self, sheet_id, data, method):
        """
        calculate values for dmdt, dmdt_sd and cumul.

        INPUTS:
            sheet_id: the identifier of the ice sheet for which data
                      is being calculated
            data: the data tuple from a GraceLoader instance
            method: either VARIABLE or FIXED - the method used to calculate
                    the dm/dt
        """
        # get time values
        t = data.date
        # set constant dmdt std. dev.
        dmdt_sd = data.date * 0. + GRACE_DMDT_SD[sheet_id]
        # get the cumulative mass change by calculating
        #   the moving average of the input mass data
        cumul = move_av(WIDTH, t, data.mass, clip=CLIP)

        if method == VARIABLE:
            # variable method - calculate dm/dt by taking the derivative
            #   of the input mass data, and perform a moving av. on the
            #   result
            dmdt = deriv_imbie(data.date, data.mass, width=WIDTH, clip=CLIP)
        else:
            # fixed method - set constant dm/dt values
            dmdt = data.date * 0. + GRACE_DMDT_FIXED[sheet_id]

        # call the base class initializer
        SourceCollection.__init__(self, t, dmdt, dmdt_sd, cumul)


class IOMData(SourceCollection):
    """
    Source collection for the input/output (racmo & rignot) data.

    This class is initialized in the same way as the base class,
    so no code is actually needed here. This class is defined simply
    to make it explicit that an instance of this class should contain
    IOM data.
    """
    pass


class ICESatData(SourceCollection):
    """
    source collection for ICESat data.
    """
    def __init__(self, data):
        """
        calc. values for cumul, and initialize the instance

        INPUTS:
            data: the data from an ICESatLoader instance
        """
        # calculate cumulative mass by taking the cumulative sum of the dm/dt
        #   values
        cumul = np.cumsum(data.dmdt) / 12.
        # call the base class initializer
        SourceCollection.__init__(self, data.t, data.dmdt, data.dmdt_sd, cumul)

import numpy as np
from collections import OrderedDict

from ..functions import ts_combine, match, get_offset, fit_imbie, lag_correlate, rmsd
from ..consts import *


class BaseSheetProcessor:
    """
    A base class from which SheetProcessor, AISProcessor and AISGrISProcessor
    inherit. This class should not be used directly, and is incomplete.
    """
    grace_dmdt_method = VARIABLE
    random_walk = True
    reconciliation_method = X4

    nsigma = 3
    verbose = False

    def __init__(self, sheet_id, **config):
        """
        basic initialization. Stores the name of the sheet, and optionally overrides
        the default configuration values

        INPUTS:
            sheet_id: the abbreviated name of the ice sheet
            **config: (optional) keyword arguments to override default config values
        """

        # update the configuration
        for k, val in config.items():
            if hasattr(self, k):
                setattr(self, k, val)
            else:
                raise ValueError("No such configuration variable: %s" % k)
        # store the sheet name
        self.sheet_id = sheet_id

        self.t = None
        self.dmdt = None
        self.dmdt_sd = None
        self.cumul = None
        self.cumul_sd = None

    def merge(self):
        """
        Performs the final steps of the data merging/accumulation process - creates
        cumulative mass and error estimates from the dm/dt data.
        """
        # calculate cumulative mass change
        self.cumul = np.cumsum(np.where(np.isfinite(self.dmdt), self.dmdt, 0)) / 12.

        # calculate standard deviation
        if self.random_walk:
            self.cumul_sd = np.cumsum(
                self.dmdt_sd / 12 / np.sqrt(np.arange(1, len(self.dmdt_sd) + 1) / 12.))
        else:
            self.cumul_sd = np.cumsum(self.dmdt_sd / 12.)

    def print_dmdt_data(self):
        """
        prints the grace dm/dt data to the terminal
        """
        x = self.t
        y = self.cumul
        y_err = self.dmdt_sd
        y_err2 = self.cumul_sd

        x0 = [1990, 1990, 1993, 2000, 2000, 2003 + 10. / 12, 2005]
        x1 = [2012, 2000, 2003, 2010, 2012, 2009, 2010]
        grace_method_names = {
            VARIABLE: 'variable',
            FIXED: 'fixed'
        }
        output_line = "  {:.02f} - {:.02f} dM/dt:{: 10.04f} +/-{:8.04f}, dM: {: 10.04f} +/-{:9.04f}, {:5.02f} yrs"

        dmdt_method = grace_method_names[self.grace_dmdt_method]
        recn_method = 'x' + self.reconciliation_method
        cerr_method = 'random walk' if self.random_walk else 'stacked'

        print self.sheet_id, "dM/dt:"
        print ' GRACE dM/dt           =', dmdt_method
        print ' reconciliation method =', recn_method
        print ' cumulative error      =', cerr_method

        for x_min, x_max in zip(x0, x1):

            ok = np.logical_and(
                np.logical_and(x >= x_min, x <= x_max),
                np.isfinite(y)
            )
            if not ok.any():
                continue

            ok = np.flatnonzero(ok)
            dm = y[ok[-1]] - y[ok[0]]
            dmdt = (y[ok[-1]] - y[ok[0]]) / (x[ok[-1]] - x[ok[0]])
            dmdt_sd = np.nanmean(y_err[ok])
            dm_sd = y_err2[ok[-1]] - y_err2[ok[0]]

            years = np.max(x[ok]) - np.min(x[ok])
            print output_line.format(x_min, x_max, dmdt, dmdt_sd, dm, dm_sd, years)

    def as_dict(self, include_sources=False):
        """
        returns the ice sheet's data as a python dict
        """
        return OrderedDict([
            ("t", self.t),
            ("dmdt", self.dmdt),
            ("dmdt_sd", self.dmdt_sd),
            ("cumul", self.cumul),
            ("cumul_sd", self.cumul_sd)
        ])

    def print_acceleration_trend(self):
        """
        Computes and prints to the terminal the estimated acceleration trend.

        The trend is found using 1st-order poly-fitting.
        """
        # get the values
        t0 = 1990
        yy = self.dmdt[:]
        xx = self.t[:] - t0

        # select only finite values
        ok = np.logical_and(
            np.isfinite(xx),
            np.isfinite(yy)
        )
        xx = xx[ok]
        yy = yy[ok]

        # compute the fit
        fit = np.polyfit(xx, yy, 1)
        yfit = np.poly1d(fit)
        # compute the error
        yerr = rmsd(yfit(xx), yy)

        # print the results
        print self.sheet_id, 'dM2/dt2 =', fit[1], '+/-', yerr, 'Gt/yr/yr'


class SheetProcessor(BaseSheetProcessor):
    """
    Inherits from BaseSheetProcessor. This class is used
    to process data from the four basic ice sheets (WAIS, EAIS, APIS and GrIS).
    """
    def __init__(self, sheet_id, data_collection, **config):
        """
        Initializes the sheet processor. In addition to the basic SheetProcessor
        init, this method also retreives data from the provided DataCollection instance.

        INPUTS:
            sheet_id: the abbreviated name of the ice sheet
            data_collection: an instance of imbie.data.DataCollection from which to
                             retrieve the data relevant to this ice sheet
            **config: (optional) keyword arguments to override default config values
        """
        # initialize the base class
        BaseSheetProcessor.__init__(self, sheet_id, **config)
        # get the grace data for this sheet
        self.grace = data_collection.get_grace(sheet_id, self.grace_dmdt_method)
        # get the icesat data for this sheet
        self.icesat = data_collection.get_icesat(sheet_id)
        # get the input/output data for this sheet
        self.iom = data_collection.get_iom(sheet_id)
        # attempt to get radar altimetry data for this sheet (if it exists)
        ra = data_collection.try_get_ra(sheet_id)
        if ra is not None:
            # if the RA data exists, store it.
            self.ra = ra

        self.cumul_grace = None
        self.cumul_icesat = None
        self.cumul_sd = None
        self.cumul_iom = None
        self.cumul_ra = None

    def as_dict(self, include_sources=False):
        """
        returns the ice sheet's data as a python dict
        """
        d = BaseSheetProcessor.as_dict(self)
        if include_sources:
            d["grace"] = self.grace.as_dict()
            d["iom"] = self.iom.as_dict()
            d["icesat"] = self.icesat.as_dict()
            if hasattr(self, 'ra'):
                d["ra"] = self.ra.as_dict()
        d["data"] = self.data.T
        d["data_sd"] = self.data_sd.T

        return d

    def print_ra_trend(self):
        """
        Fits linear trends to the radar altimetry data, and
        prints the outputs
        """
        if not hasattr(self, 'ra'):
            raise ValueError("No RA data for this ice sheet")

        x_min = 2003 + 10. / 12
        x_max = 2009.
        x_range = (x_min, x_max)

        _, fit, err = fit_imbie(self.ra.t, self.ra.cumul, x_range=x_range, width=13./12, full=True)
        ok = np.logical_and(self.ra.t >= x_min,
                            self.ra.t <= x_max)
        output_line = "{:7.02f} - {:7.02f}: ({}) +/- {}"
        print "dM/dt(RA)", self.sheet_id, output_line.format(x_min, x_max, fit, np.mean(err[ok]))

    def print_lag_correlation(self):
        """
        Performs cross-correlation on the input sources, and prints the results.
        """
        if not hasattr(self, 'ra'):
            raise ValueError("No RA data for this ice sheet")

        c_ra_grace, lag_ra_grace = \
            lag_correlate(self.ra.t, self.ra.dmdt, self.grace.t, self.grace.dmdt, return_lag=True)
        c_ra_iom, lag_ra_iom = \
            lag_correlate(self.ra.t, self.ra.dmdt, self.iom.t, self.iom.dmdt, return_lag=True)
        c_iom_grace, lag_iom_grace = \
            lag_correlate(self.iom.t, self.iom.dmdt, self.grace.t, self.grace.dmdt, return_lag=True)

        print self.sheet_id
        print '  r^2(c_iom_grace)', c_iom_grace ** 2., lag_iom_grace
        print '  r^2(c_ra_grace)', c_ra_grace ** 2., lag_ra_grace
        print '  r^2(c_ra_iom)', c_ra_iom ** 2., lag_ra_iom

    def merge(self):
        """
        Merge the data from the different input sources to create overall
        estimations for this ice sheet.
        """
        # if this sheet has radar data...
        if hasattr(self, 'ra'):
            # ... use the appropriate merge method
            self._merge_ra()
        else:
            # ... otherwise, use the non-RA method
            self._merge_no_ra()
        # perform the accumulation via the base class
        BaseSheetProcessor.merge(self)

        # calculate cumulative results for each of the input sources
        self.cumul_iom = \
            get_offset(self.t, self.cumul, self.iom.t, self.iom.cumul)
        self.cumul_grace = \
            get_offset(self.t, self.cumul, self.grace.t, self.grace.cumul)
        self.cumul_icesat = \
            get_offset(self.t, self.cumul, self.icesat.t, self.icesat.cumul)
        if hasattr(self, 'ra'):
            self.cumul_ra = \
                get_offset(self.t, self.cumul, self.ra.t, self.ra.cumul)

    def _merge_ra(self):
        """
        performs data merging, if the SheetProcessor instance has
        radar altimetry data in addition to the other sources.

        calculates overall dm/dt and error values from each of the
        input sources.
        """
        if self.reconciliation_method == X3:
            # x3 mode
            # merge dm/dt
            t_ra_la, dmdt_ra_la = \
                ts_combine([self.ra.t, self.icesat.t],
                           [self.ra.dmdt, self.icesat.dmdt],
                           nsigma=self.nsigma, verbose=self.verbose)
            self.t, self.dmdt, self.data = \
                ts_combine([self.iom.t, self.grace.t, t_ra_la],
                           [self.iom.dmdt, self.grace.dmdt, dmdt_ra_la],
                           nsigma=self.nsigma, verbose=self.verbose, ret_data_out=True)
            # merge dm/dt sd
            t_ra_la, dmdt_sd_ra_la = \
                ts_combine([self.ra.t, self.icesat.t],
                           [self.ra.dmdt_sd, self.icesat.dmdt_sd],
                           nsigma=self.nsigma, verbose=self.verbose)
            self.t, self.dmdt_sd, self.data_sd = \
                ts_combine([self.iom.t, self.grace.t, t_ra_la],
                           [self.iom.dmdt_sd, self.grace.dmdt_sd, dmdt_sd_ra_la],
                           nsigma=self.nsigma, verbose=self.verbose, ret_data_out=True)
        else:
            # x4 mode
            # merge dm/dt
            self.t, self.dmdt, self.data = \
                ts_combine([self.iom.t, self.grace.t, self.icesat.t, self.ra.t],
                           [self.iom.dmdt, self.grace.dmdt, self.icesat.dmdt, self.ra.dmdt],
                           nsigma=self.nsigma, verbose=self.verbose, ret_data_out=True)
            # merge dm/dt sd
            self.t, self.dmdt_sd, self.data_sd = \
                ts_combine([self.iom.t, self.grace.t, self.icesat.t, self.ra.t],
                           [self.iom.dmdt_sd, self.grace.dmdt_sd, self.icesat.dmdt_sd, self.ra.dmdt_sd],
                           nsigma=self.nsigma, verbose=self.verbose, ret_data_out=True)

    def _merge_no_ra(self):
        """
        performs data merging, if the SheetProcessor instance does not
        have radar altimetry data.

        calculates overall dm/dt and error values from each of the
        input sources.
        """
        # merge dm/dt
        self.t, self.dmdt, self.data = \
            ts_combine([self.iom.t, self.grace.t, self.icesat.t],
                       [self.iom.dmdt, self.grace.dmdt, self.icesat.dmdt],
                       nsigma=self.nsigma, verbose=self.verbose, ret_data_out=True)
        # merge dm/dt sd
        self.t, self.dmdt_sd, self.data_sd = \
            ts_combine([self.iom.t, self.grace.t, self.icesat.t],
                       [self.iom.dmdt_sd, self.grace.dmdt_sd, self.icesat.dmdt_sd],
                       nsigma=self.nsigma, verbose=self.verbose, ret_data_out=True)

    def plot_dmdt(self, plotter):
        """
        creates a dm/dt + error vs time plot for this ice sheet.

        INPUTS:
            plotter: an imbie.plotting.Plotter instance which will render the plots
        """
        # set rendering order
        o = ['ICESat', 'IOM', 'RA', 'GRACE']
        # collect time values into a dictionary
        t = {
            "ICESat": self.icesat.t,
            "IOM": self.iom.t,
            "GRACE": self.grace.t
        }
        # collect dm/dt data
        dmdt = {
            "ICESat": self.icesat.dmdt,
            "IOM": self.iom.dmdt,
            "GRACE": self.grace.dmdt
        }
        # collect standard deviation data
        dmdt_sd = {
            "ICESat": self.icesat.dmdt_sd,
            "IOM": self.iom.dmdt_sd,
            "GRACE": self.grace.dmdt_sd
        }
        if hasattr(self, 'ra'):
            t['RA'] = self.ra.t
            dmdt['RA'] = self.ra.dmdt
            dmdt_sd['RA'] = self.ra.dmdt_sd
        else:
            o.remove('RA')
        # create the plot
        plotter.plot_dmdt_sheet(
            self.sheet_id, t, dmdt, dmdt_sd, order=o
        )

    def plot_mass(self, plotter):
        """
        creates a mass vs time plot from the ice sheet's data.

         INPUT:
            plotter: an imbie.plotting.Plotter instance which will render the plots
        """
        t = {
            "ICESat": self.icesat.t,
            "IOM": self.iom.t,
            "GRACE": self.grace.t,
            "All": self.t
        }
        cumul = {
            "ICESat": self.icesat.cumul,
            "IOM": self.iom.cumul,
            "GRACE": self.grace.cumul,
            "All": self.cumul
        }
        if hasattr(self, 'ra'):
            t['RA'] = self.ra.t
            cumul['RA'] = self.ra.cumul

        plotter.plot_ice_mass(self.sheet_id, t, cumul)


class AISProcessor(BaseSheetProcessor):
    """
    Inherits from BaseSheetProcessor. This class is designed to calculate
    overall Antarctic data by amalgamating data from the three Antarctic
    ice sheets.
    """
    def __init__(self, wais, eais, apis, **config):
        """
        Initializes the AISProcessor instance.

        INPUTS:
            wais: West Antarctica SheetProcessor instance
            eais: East Antarctica SheetProcessor instance
            apis: Antarctic Peninsula SheetProcessor instance
            **config: (optional) keyword arguments to override default config values
        """
        # initialize the base class
        BaseSheetProcessor.__init__(self, "AIS", **config)

        self.apis = apis
        self.wais = wais
        self.eais = eais

        # set time values
        self.t = apis.t
        # sum the dm/dt values
        self.dmdt = wais.dmdt + eais.dmdt + apis.dmdt
        # calc. overall standard deviation
        self.dmdt_sd = np.sqrt(
            wais.dmdt_sd ** 2. + eais.dmdt_sd ** 2. + apis.dmdt_sd ** 2.
        )


class AISGrISProcessor(BaseSheetProcessor):
    """
    Inherits from BaseSheetProcessor. This class is designed to calculate
    overall Antarctica & Greenland data by amalgamating data from an
    AISProcessor instance and the Greenland SheetProcessor instance.
    """
    def __init__(self, ais, gris, **config):
        """
        Initializes the Processor.

        INPUTS:
            ais: the AISProcessor instance
            gris: the Greenland SheetProcessor instance
            **config: (optional) keyword arguments to override default config values
        """
        # initialize the base class
        BaseSheetProcessor.__init__(self, "AIS+GrIS", **config)

        # find indices at which the time values from each processor match
        m1, m2 = match(ais.t, gris.t)
        # get the time values at those indices
        self.t = ais.t[m1]
        # sum the dm/dt values
        self.dmdt = ais.dmdt[m1] + gris.dmdt[m2]
        # calc. standard deviation
        self.dmdt_sd = np.sqrt(ais.dmdt_sd[m1] ** 2. + gris.dmdt_sd[m2] ** 2.)
        # calc. the cumulative mass change.
        self.cumul = ais.cumul[m1] + gris.cumul[m2]

    def merge(self):
        """
        Performs the final steps of the data merging/accumulation process - creates
        cumulative mass and error estimates from the dm/dt data.
        """
        # calculate standard deviation
        if self.random_walk:
            self.cumul_sd = np.cumsum(
                self.dmdt_sd / 12 / np.sqrt(np.arange(1, len(self.dmdt_sd) + 1) / 12.))
        else:
            self.cumul_sd = np.cumsum(self.dmdt_sd / 12.)
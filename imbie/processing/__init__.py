from .sheet_processor import SheetProcessor, AISProcessor, AISGrISProcessor
from ..data.output import save_data
import os

__author__ = 'Mark'

class Processor:
    """
    This class performs the data processing for IMBIE 2.

    it collects input data from a DataCollection instance,
    and uses this to create a SheetProcessor instance for each
    ice sheet (or collection of ice sheets). Each SheetProcessor
    then performs merging and accumulation on its data.

    The Processor instance can then output the computed data, or
    render graphs of the results via a Plotter instance.
    """

    def __init__(self, data, **config):
        """
        initializes the Processor instance.

        INPUTS:
            data: an instance of the imbie.data.DataCollection class via which
                  this Processor will load the input data.
            **config: (optional) provide keyword arguments to override default
                      configurations on each SheetProcessor instance.
        """
        # store the input parameters
        self.config = config
        self.data = data

        self.wais = None
        self.eais = None
        self.apis = None
        self.gris = None
        self.ais = None
        self.ais_gris = None

    def __iter__(self):
        """
        iterates through the processor's sheet processors
        """
        yield self.wais
        yield self.eais
        yield self.apis
        yield self.gris
        yield self.ais
        yield self.ais_gris

    def assign(self):
        """
        roughly equivalent to the 'ASSIGN' section of the IMBIE 1 IDL source code,
        this method instructs the DataCollection instance to read all the required
        files, and then creates SheetProcessor instances from this data for each of
        the four basic ice sheets.
        """
        # read the input data
        self.data.read()
        # create SheetProcessor instances
        self.wais = SheetProcessor("WAIS", self.data, **self.config)
        self.eais = SheetProcessor("EAIS", self.data, **self.config)
        self.apis = SheetProcessor("APIS", self.data, **self.config)
        self.gris = SheetProcessor("GrIS", self.data, **self.config)

    def merge(self):
        """
        roughly equivalent the 'merge' section of the original IMBIE 1 IDL code,
        this method instructs each of the four basic SheetProcessor instances to
        merge & accumulate their data, and then creates the ice sheet collection
        instances from the merged data.
        """
        # perform merging
        self.wais.merge()
        self.eais.merge()
        self.apis.merge()
        self.gris.merge()
        # create overall Antarctic data
        self.ais = AISProcessor(self.wais, self.eais, self.apis, **self.config)
        self.ais.merge()
        # create Antarctic + Greenland data
        self.ais_gris = AISGrISProcessor(self.ais, self.gris, **self.config)
        self.ais_gris.merge()

    def print_stats(self):
        """
        prints various statistics to the terminal
        """
        for sheet in self.wais, self.eais, self.apis, self.gris:
            sheet.print_dmdt_data()
        for sheet in self.wais, self.eais:
            sheet.print_ra_trend()
        for sheet in self.wais, self.eais:
            sheet.print_lag_correlation()
        for sheet in self.wais, self.eais, self.apis, self.gris, self.ais:
            sheet.print_acceleration_trend()

    def save(self, path):
        """
        Outputs the calculated results to a file.

        INPUTS:
            path: the path to the output folder
        """
        if not os.path.exists(path):
            os.mkdir(path)
        for sheet in self.wais, self.eais, self.apis, self.gris:
            rcon_method = sheet.reconciliation_method
            fname = "imbie_all_{}.x{}.csv".format(sheet.sheet_id.lower(), rcon_method)

            fpath = os.path.join(path, fname)
            data = sheet.as_dict()

            save_data(data, fpath)

    def plot_dmdt(self, plotter):
        """
        Creates dmdt + error vs time plots for each of the four basic ice sheets.

        INPUTS:
            plotter: an imbie.plotting.Plotter instance which will render the plots
        """

        self.wais.plot_dmdt(plotter)
        self.eais.plot_dmdt(plotter)
        self.apis.plot_dmdt(plotter)
        self.gris.plot_dmdt(plotter)

    def plot_mass(self, plotter):
        """
        Creates mass vs time plots for each of the four basic ice sheets

        INPUTS:
            plotter: an imbie.plotting.Plotter instance which will render the plots
        """
        self.wais.plot_mass(plotter)
        self.eais.plot_mass(plotter)
        self.apis.plot_mass(plotter)
        self.gris.plot_mass(plotter)

    def plot_cumulative(self, plotter):
        """
        Creates mass + error vs time plots for two groups of ice sheets / sheet collections

        INPUTS:
            plotter: an imbie.plotting.Plotter instance which will render the plots
        """
        # plot the Antarctic ice sheets
        self._plot_cumulative_sheets(plotter, self.wais, self.eais, self.apis)
        # plot the Antarctic collection, Greenland, and the combined Antarctic + Greenland collection
        self._plot_cumulative_sheets(plotter, self.ais, self.gris, self.ais_gris)

    @staticmethod
    def _plot_cumulative_sheets(plotter, *sheets):
        """
        static method used by the Processor.plot_cumulative method.

        INPUTS:
            plotter: an imbie.plotting.Plotter instance which will render the plots
            *sheets: the SheetProcessor instances to plot the data from
        """
        t = {}
        cumul = {}
        cumul_sd = {}
        for sheet in sheets:
            t[sheet.sheet_id] = sheet.t
            cumul[sheet.sheet_id] = sheet.cumul
            cumul_sd[sheet.sheet_id] = sheet.cumul_sd
        plotter.plot_reconciled(t, cumul, cumul_sd)

    def plot_all(self, plotter):
        """
        render all the plots of the computed data.

        INPUTS:
            plotter: an imbie.plotting.Plotter instance which will render the plots
        """
        self.plot_dmdt(plotter)
        self.plot_cumulative(plotter)
        self.plot_mass(plotter)
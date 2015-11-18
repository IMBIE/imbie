import unittest
import os
import numpy as np

from imbie.plotting import Plotter


class TestPlotting(unittest.TestCase):
    """
    class for testing the Plotter class
    """
    def setUp(self):
        self.folder = os.path.join("testdata", "outputs")
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        else:
            used_names = [
                'dmdt_ice_sheet_method.png', 'dmdt_ice_sheet_test.png',
                'reconciled_mass_eais_wais_apis.png', 'mass_ice_sheet_test.png'
            ]
            for fname in used_names:
                path = os.path.join(self.folder, fname)
                if os.path.exists(path):
                    os.unlink(path)

        self.plotter = Plotter(file_type='png', path=self.folder)

    def test_box_plotter(self):
        self.plotter.plot_03_08_dmdt_ice_sheet_method()

        opath = os.path.join(self.folder, 'dmdt_ice_sheet_method.png')
        exists = os.path.exists(opath)

        self.assertTrue(exists, "Box plot image exists")
        self.assertGreater(os.path.getsize(opath), 0)

    def test_dmdt_plotter(self):
        sources = ["ICESat", "GRACE", "RA", "IOM"]
        t = {src: np.linspace(1992, 2012, 300) for src in sources}
        dmdt = {src: -np.cumsum(np.random.random([300])) for src in sources}
        dmdt_sd = {src: np.ones([300]) * 30 for src in sources}

        self.plotter.plot_dmdt_sheet("Test", t, dmdt, dmdt_sd, sources)

        opath = os.path.join(self.folder, 'dmdt_ice_sheet_test.png')
        exists = os.path.exists(opath)

        self.assertTrue(exists, "dmdt plot image exists")
        self.assertGreater(os.path.getsize(opath), 0)

    def test_reconciled_plotter(self):
        sheets = ["EAIS", "WAIS", "APIS"]
        t = {src: np.linspace(1992, 2012, 300) for src in sheets}
        dmdt = {src: -np.cumsum(np.random.random([300])) for src in sheets}
        dmdt_sd = {src: np.ones([300]) * 30 for src in sheets}

        self.plotter.plot_reconciled(t, dmdt, dmdt_sd, sheets)

        opath = os.path.join(self.folder, 'reconciled_mass_eais_wais_apis.png')
        exists = os.path.exists(opath)

        self.assertTrue(exists, "recon plot image exists")
        self.assertGreater(os.path.getsize(opath), 0)

    def test_mass_plotter(self):
        sources = ["ICESat", "GRACE", "RA", "IOM"]
        t = {src: np.linspace(1992, 2012, 300) for src in sources}
        mass = {src: -np.cumsum(np.random.random([300])) for src in sources}

        self.plotter.plot_ice_mass("Test", t, mass, sources)

        opath = os.path.join(self.folder, 'mass_ice_sheet_test.png')
        exists = os.path.exists(opath)

        self.assertTrue(exists, "mass plot image exists")
        self.assertGreater(os.path.getsize(opath), 0)

if __name__ == "__main__":
    unittest.main()

import unittest
import os
import numpy as np

from imbie.data.loaders import *


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.join("testdata", "inputs")

    def test_dm_loader(self):
        dm = DMLoader(root=self.folder, fname='dm.xml')
        out = dm.read()

        expected = 1.4907072147138756

        self.assertAlmostEqual(out.WAIS, expected)
        self.assertAlmostEqual(out.APIS, expected)
        self.assertAlmostEqual(out.EAIS, expected)

    def test_grace_loader(self):
        grace = GraceLoader(
            root=self.folder, folder="",
            APIS='grace.csv',
            EAIS='grace.csv',
            WAIS='grace.csv',
            GrIS='grace.csv',
            AIS='grace.csv'
        )
        out = grace.read()

        expected_month = np.array([00, 01, 02, 03])
        expected_mass = np.array([10, 11, 12, 13])
        expected_date = np.array([20, 21, 22, 23])
        expected_ngroups = np.array([30, 31, 32, 33])

        for sheet in out:
            self.assertTrue(np.allclose(sheet.month, expected_month))
            self.assertTrue(np.allclose(sheet.mass, expected_mass))
            self.assertTrue(np.allclose(sheet.date, expected_date))
            self.assertTrue(np.allclose(sheet.ngroups, expected_ngroups))

    def test_icesat_loader(self):
        icesat = ICESatLoader(root=self.folder, fname='icesat.xml')
        out = icesat.read()

        self.assertTrue(np.allclose(out.EAIS.dmdt, 1))
        self.assertTrue(np.allclose(out.EAIS.dmdt_sd, 2))
        self.assertTrue(np.allclose(out.WAIS.dmdt, 3))
        self.assertTrue(np.allclose(out.WAIS.dmdt_sd, 4))
        self.assertTrue(np.allclose(out.APIS.dmdt, 5))
        self.assertTrue(np.allclose(out.APIS.dmdt_sd, 6))

        expected_dmdt = np.linspace(1, 4, 9*12 + 1)[:-1]
        self.assertTrue(np.allclose(out.GrIS.dmdt, expected_dmdt))
        self.assertTrue(np.allclose(out.GrIS.dmdt_sd, 5))

    def test_ra_loader(self):
        ra = RALoader(
            root=self.folder, mm_folder="",
            mm_fname="mm.csv", avs_folder="",
            EAIS="avs_eais.txt",
            WAIS="avs_wais.txt"
        )

        out = ra.read()

        expected_time = np.array([2000., 2000.5, 2001., 2001.5, 2002.])

        self.assertTrue(np.allclose(out.EAIS.time, expected_time))
        self.assertTrue(np.allclose(out.WAIS.time, expected_time))
        self.assertTrue(np.allclose(out.EAIS.dM_sd, 39.6444))
        self.assertTrue(np.allclose(out.WAIS.dM_sd, 0))

    def test_racmo_loader(self):
        racmo = RacmoLoader(root=self.folder, folder='', fpattern='racmo')
        out = racmo.read()

        expected_time = np.linspace(2000, 2001, 13)[1:]
        self.assertTrue(np.allclose(out.time, expected_time))

    def test_rignot_loader(self):
        path = os.path.join(self.folder, 'rignot.csv')

        rignot = RignotLoader(root=self.folder, folder='', fname='rignot.csv')
        out = rignot.read()

        self.assertTrue(
            np.allclose(out.DATE_IOM,
                        np.linspace(1992, 1992.75, 10))
        )
        self.assertTrue(
            np.allclose(out.GrIS.SMB, 400)
        )
        self.assertTrue(
            np.allclose(out.GrIS.SMB_s, 60)
        )
        self.assertTrue(
            np.allclose(out.GrIS.D, 500)
        )
        self.assertTrue(
            np.allclose(out.GrIS.D_s, 26)
        )
        self.assertTrue(
            np.allclose(out.GrIS.TMB, 0)
        )
        self.assertTrue(
            np.allclose(out.GrIS.TMB_s, 65)
        )
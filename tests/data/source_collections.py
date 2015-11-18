from imbie.data.source_collections import *
from imbie.data.loaders.ra import RAMMIS
from imbie.data.loaders.grace import GraceIS
from imbie.data.loaders.icesat import ICESatIS

import unittest
import numpy as np

class TestSourceCollections(unittest.TestCase):
    def test_ra_data(self):
        in_data = RAMMIS(
            time=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            dM=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
            dM_sd=np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
        )
        collection = RAData(in_data, 0)

        ok = np.isfinite(collection.dmdt)
        self.assertTrue(np.allclose(collection.dmdt[ok], 0))
        self.assertTrue(np.allclose(collection.dmdt_sd, 6))
        self.assertTrue(np.allclose(collection.cumul, 2))

    def test_grace_data(self):
        in_data = GraceIS(
            month=np.array([1, 2, 3, 4, 5]),
            mass=np.array([10, 20, 30, 40, 50]),
            date=np.array([1, 2, 3, 4, 5]),
            ngroups=np.array([0, 0, 0, 0, 0])
        )

        collection = GRACEData("WAIS", in_data, 'v')

        ok = np.isfinite(collection.dmdt)
        self.assertTrue(np.allclose(collection.dmdt[ok], 10))

    def test_icesat_data(self):
        in_data = ICESatIS(
            t=np.arange(1, 6),
            dmdt=np.ones([5]) * 5,
            dmdt_sd=np.zeros([5]),
        )
        collection = ICESatData(in_data)

        self.assertTrue(
            np.allclose(collection.cumul,
                        np.array([0.41666667, 0.83333333, 1.25, 1.66666667, 2.08333333]))
        )

if __name__ == "__main__":
    unittest.main()
from imbie.data.output import save_data

import numpy as np
import unittest
import os
from collections import OrderedDict
import csv

class OutputTests(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.join("testdata", "outputs")
        self.fname = "output.csv"
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        else:
            path = os.path.join(self.folder, self.fname)
            if os.path.exists(path):
                os.unlink(path)

    def test_output(self):
        data = OrderedDict()
        data["ones"] = np.ones([3])
        data["zeros"] = np.zeros([3, 10])
        data["twos"] = np.ones([3]) * 2.

        path = os.path.join(self.folder, self.fname)
        save_data(data, path)

        self.assertTrue(os.path.exists(path))

        header = True
        expected_headers = ["ones", "zeros_0", "zeros_1", "zeros_2", "zeros_3",
                            "zeros_4", "zeros_5", "zeros_6", "zeros_7", "zeros_8",
                            "zeros_9", "twos"]
        expected_values = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.]

        with open(path, 'rb') as f:
            r = csv.reader(f)
            for line in r:
                if header:
                    self.assertSequenceEqual(line, expected_headers)
                    header = False
                else:
                    self.assertTrue(
                        np.allclose([float(n) for n in line], expected_values)
                    )


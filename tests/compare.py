from imbie.data.loaders.base import DataLoader
from imbie.functions import rmsd, match

import os
import matplotlib.pyplot as plt
import numpy as np
import sys


class OutputLoader(DataLoader):
    def __init__(self, root, fname, delim):
        DataLoader.__init__(self)

        path = os.path.join(root, fname)
        self.data = self.read_csv(
            path, delim=delim, header=2
        )
        self.t = self.data[0, :]
        self.dmdt = self.data[1, :]
        self.dmdt_sd = self.data[2, :]
        self.dm = self.data[3, :]
        self.dm_sd = self.data[4, :]


def compare_outputs(folder, file1, file2, ):
    a = OutputLoader(folder, file1, ' ')
    b = OutputLoader(folder, file2, ',')

    plt.plot(a.t, a.dmdt,
             b.t, b.dmdt)
    plt.show()

    ia, ib = match(a.t, b.t)
    print "dM/dt:   ", rmsd(a.dmdt[ia], b.dmdt[ib])
    print "dM/dt sd:", rmsd(a.dmdt_sd[ia], b.dmdt_sd[ib])
    print "dM:      ", rmsd(a.dm[ia], b.dm[ib])
    print "dM_sd:   ", rmsd(a.dm_sd[ia], b.dm_sd[ib])


if __name__ == "__main__":
    compare_outputs(*sys.argv[-3:])
import numpy as np

from imbie2.model.collections import WorkingMassRateCollection
from imbie2.model.series import WorkingMassRateDataSeries

from .functions import ts2m
import matplotlib.pyplot as plt


def count_tolerance(data: WorkingMassRateCollection, guide: WorkingMassRateDataSeries, nsigma: int=1) -> np.ndarray:
    """
    create series counting number of contributions within
    nsigma tolerance of the guide series at each epoch
    """
    counts = np.zeros(guide.dmdt.shape, dtype=np.int)
    tot = np.zeros_like(counts)

    for series in data:      
        i_g = np.logical_and(
            guide.t >= series.t.min(),
            guide.t <= series.t.max()
        )

        series_dmdt = np.interp(guide.t[i_g], series.t, series.dmdt)
        series_errs = np.interp(guide.t[i_g], series.t, series.errs)      

        diffs = np.abs(guide.dmdt[i_g] - series_dmdt)

        margin = np.sqrt(series_errs ** 2. + guide.errs[i_g] **2.)

        within = diffs < margin * nsigma

        counts[i_g] += within
        tot[i_g] += 1

    return guide.t, counts, tot

import numpy as np
from numpy.linalg import solve, qr, cholesky
from typing import Tuple

from imbie2.const.lsq_methods import LSQMethod
from imbie2.util.functions import ts2m


def lscov(a, b, v: np.ndarray = None, dx: bool = False) -> np.ndarray:
    """
    This is a python implementation of the matlab lscov function. This has been written based upon the matlab source
    code for lscov.m, which can be found here: http://opg1.ucsd.edu/~sio221/SIO_221A_2009/SIO_221_Data/Matlab5/Toolbox/matlab/matfun/lscov.m
    """

    m, n = a.shape
    if m < n:
        raise Exception(f"problem must be over-determined so that M > N. ({m}, {n})")
    if v is None:
        v = np.eye(m)

    if v.shape != (m, m):
        raise Exception("v must be a {0}-by-{0} matrix".format(m))

    qnull, r = qr(a, mode="complete")
    q = qnull[:, :n]
    r = r[:n, :n]

    qrem = qnull[:, n:]
    g = qrem.T.dot(v).dot(qrem)
    f = q.T.dot(v).dot(qrem)

    c = q.T.dot(b)
    d = qrem.T.dot(b)

    x = solve(r, (c - f.dot(solve(g, d))))

    # This was not required for merge_dM, and so has been removed as it has problems.
    if dx:
        u = cholesky(v).T
        z = solve(u, b)
        w = solve(u, a)
        mse = (z.T.dot(z) - x.T.dot(w.T.dot(z))) / (m - n)
        q, r = qr(w)
        ri = solve(r, np.eye(n)).T
        dx = np.sqrt(np.sum(ri * ri, axis=0) * mse).T

        return x, dx
    return x


def dm_to_dmdt(
    t: np.ndarray,
    dm: np.ndarray,
    sigma_dm: np.ndarray,
    wsize: float,
    tout: np.ndarray = None,
    truncate: bool = True,
    debug: bool = False,
    lsq_method: LSQMethod = LSQMethod.normal,
    tapering: bool = False,
    min_tapering: float = .75,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    create dmdt time series from dm data
    """
    # prevent user from applying conflicting options
    assert not (truncate and tapering), "conflicting options specified"

    # output postings are optional, produce records at
    #  same postings as input if not provided
    if tout is None:
        tout = t

    # create empty data structures
    dmdt = np.empty(tout.shape, dtype=t.dtype) * np.nan
    sigma_dmdt = np.empty(tout.shape, dtype=t.dtype) * np.nan
    model_fit_t = [None for _ in tout]
    model_fit_dm = [None for _ in tout]
    n_records_fitting = [0 for _ in tout]

    # size of half the window
    w_halfsize = wsize / 2.0

    # get start and end times of input series
    tmin = np.min(t)
    tmax = np.max(t)

    # iterate accross time series and apply window fitting
    for i, it in enumerate(tout):
        if it < tmin or it > tmax:
            continue

        # calc. extents of fitting window
        wmin = it - w_halfsize
        wmax = it + w_halfsize

        # check if window overlaps data boundaries
        if wmin < tmin or wmax > tmax:

            if truncate:
                # truncate mode: clip the output series
                #  to avoid producing values from incomplete
                #  windows
                dmdt[i] = np.nan
                sigma_dmdt[i] = np.nan
                continue

            elif tapering:
                # tapering mode: reduce size of window
                #  near the ends of the time series in
                #  order to maintain a symetrical window.
                #  we later replace the dmdt values at the
                #  first and last 6-month postings with the
                #  mean values for that year
                w_tapered = min(tout.max() - it, it - tout.min())
                if min_tapering is not None and w_tapered < min_tapering:
                    w_tapered = min_tapering
                wmin = it - w_tapered
                wmax = it + w_tapered

                in_window, *_ = np.logical_and(t >= wmin, t < wmax).nonzero()

                if in_window.size < 2:
                    dmdt[i] = np.nan
                    sigma_dmdt[i] = np.nan
                    continue

            else:
                # otherwise get the portion of the window that is valid
                trunc_wmin = max(wmin, tmin)
                trunc_wmax = min(wmax, tmax)

                in_window = np.logical_and(t >= trunc_wmin, t < trunc_wmax).nonzero()
        else:
            # create index array for current window
            in_window = np.logical_and(t >= wmin, t < wmax).nonzero()

        # get time, dm and error values within the current window
        window_t = t[in_window]
        window_dm = dm[in_window]
        window_sigma_dm = sigma_dm[in_window]

        # prepare input for fitting
        lsq_fit = np.vstack([np.ones_like(window_t), window_t]).T

        if lsq_method == LSQMethod.regress:
            # this option is not available
            raise NotImplementedError()
        elif lsq_method == LSQMethod.normal:
            # normal LSQ fitting method
            lsq_coef, lsq_se = lscov(lsq_fit, window_dm, dx=True)
        elif lsq_method == LSQMethod.weighted:
            # error-weighted LSQ fitting
            w = np.diag(1.0 / np.square(window_sigma_dm))
            lsq_coef, lsq_se = lscov(lsq_fit, window_dm, w, dx=True)

        # get RMS of input dm errors within window
        avg_window_sigma = np.sqrt(np.nanmean(window_sigma_dm ** 2.0))

        # get output dmdt and error values
        dmdt[i] = lsq_coef[1]
        sigma_dmdt[i] = np.sqrt(lsq_se[1] ** 2 + avg_window_sigma ** 2)

        # store data for debugging plot
        model_fit_t[i] = np.r_[wmin:wmax:0.2]
        model_fit_dm[i] = lsq_coef[0] + lsq_coef[1] * model_fit_t[i]
        n_records_fitting[i] = window_t.size

    if tapering:
        if debug:
            import matplotlib.pyplot as plt
            plt.errorbar(tout, dmdt, yerr=sigma_dmdt, label="original")
        # 6-month period to overwrite values
        i_overwrite = tout <= tout.min() + 0.5
        # 12-month period from which to calculate
        #  averages
        i_average = tout <= tout.min() + 1.0

        # calc. averages
        mean_dmdt = np.nanmean(dmdt[i_average])
        mean_sigma = np.nanmean(sigma_dmdt[i_average])

        # apply values
        dmdt[i_overwrite] = mean_dmdt
        sigma_dmdt[i_overwrite] = mean_sigma

        # 6-month period to overwrite values
        i_overwrite = tout >= tout.max() - 0.5
        # 12-month period from which to calculate
        #  averages
        i_average = tout >= tout.max() - 1.0

        # calc. averages
        mean_dmdt = np.nanmean(dmdt[i_average])
        mean_sigma = np.nanmean(sigma_dmdt[i_average])

        # apply values
        dmdt[i_overwrite] = mean_dmdt
        sigma_dmdt[i_overwrite] = mean_sigma

        if debug:
            plt.errorbar(tout, dmdt, yerr=sigma_dmdt, label="end-corrected")
            plt.legend()
            plt.show()

    if debug:
        # create debugging plots
        import matplotlib.pyplot as plt

        # ax = plt.subplot(211)
        _, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.set_xlim(np.min(tout), np.max(tout))

        t_m, dmdt_m = ts2m(tout, dmdt)
        ok = ~np.isnan(dmdt_m)
        base = np.interp(t_m[ok].min(), t, dm)
        ax1.plot(
            t_m[ok],
            base + np.cumsum(dmdt_m[ok] / 12),
            "r-",
            lw=2,
            label="cumulative dM/dt",
        )

        add_label = True
        for i in range(1, len(t), 5):
            if model_fit_t[i] is None or model_fit_dm[i] is None:
                continue
            if add_label:
                ax1.plot(
                    model_fit_t[i], model_fit_dm[i], "--k", lw=0.5, label="dM fitting"
                )
                add_label = False
            else:
                ax1.plot(model_fit_t[i], model_fit_dm[i], "--k", lw=0.5)
        for i in range(len(t)):
            ax1.annotate("%i" % n_records_fitting[i], (t[i], dm[i]))
        ax1.plot(t, dm, ".", label="original dM")

        ax1.legend()

        # ax = plt.subplot(212)
        ax2.set_xlim(np.min(tout), np.max(tout))
        ax2.errorbar(tout, dmdt, yerr=sigma_dmdt, color="b", label="dM/dt series")
        ax2.legend()

        plt.show()

    return tout, dmdt, sigma_dmdt

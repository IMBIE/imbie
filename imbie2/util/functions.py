from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate, stats, optimize
import math


def match(a, b, epsilon=0):
    """
    Python implementation of the IDL match.pro procedure

    returns lists of indices at which 'a' and 'b' match.
    e.g:
    >>> a = [3, 5, 7, 9, 11]
    >>> b = [5, 6, 7, 8, 9, 10]
    >>> ai, bi = match(a, b)
    >>> print ai
    [1, 2, 3]
    >>> print bi
    [0, 2, 4]
    >>> print a[ai]
    [5, 7, 9]
    >>> print b[bi]
    [5, 7, 9]

    N.B: This assumes that there are no duplicate values in 'a' or 'b'.
         the results will be meaningless if duplicates exist.

    TODO:
        The IDL implementation use a histogram-based method when both
        'a' and 'b' are integer arrays. This method is faster for small
        arrays. It may be worthwhile implementing this method in Python.

        However, this histogram method relies on use of the 'reverse_indices'
        argument which does not exist in numpy.

    INPUTS:
        a: the first of the two series to match
        b: the second series to match
        epsilon: (optional) the tolerance within which to consider
                 a pair of values to be a match.
    OUTPUTS:
        suba: The indices at which values in 'a' match a value in 'b'
        subb: The indices at which values in 'b' match a value in 'a'
    """
    na = len(a)
    nb = len(b)
    suba = np.zeros(a.shape, dtype=bool)
    subb = np.zeros(b.shape, dtype=bool)

    if na == 1 or nb == 1:
        if nb > 1:
            subb[:] = (b == a[0])
            if subb.any():
                suba = np.zeros((np.count_nonzero(subb)), dtype=np.int32)
            else:
                suba = np.array([])
            subb = np.flatnonzero(subb)
        else:
            suba[:] = (a == b[0])
            if suba.any():
                subb = np.zeros((np.count_nonzero(suba)), dtype=np.int32)
            else:
                subb = np.array([])
            suba = np.flatnonzero(suba)
        return suba, subb
    c = np.concatenate((a, b))
    ind = np.concatenate(
        [np.arange(na), np.arange(nb)]
    )
    vec = np.concatenate(
        [np.zeros([na], dtype=bool),
         np.ones([nb], dtype=bool)]
    )

    sub = np.argsort(c)
    c = c[sub]
    ind = ind[sub]
    vec = vec[sub]

    # n = na + nb
    if epsilon == 0:
        firstdup = np.logical_and(
            c == np.roll(c,   -1),
            vec != np.roll(vec, -1)
        )
    else:
        firstdup = np.logical_and(
            np.abs(c - np.roll(c, -1)) < epsilon,
            vec != np.roll(vec, -1)
        )
    count = np.count_nonzero(firstdup)
    firstdup = np.flatnonzero(firstdup)
    if count == 0:
        suba = np.array([])
        subb = np.array([])
        return suba, subb
    dup = np.zeros((count * 2), dtype=int)
    even = np.arange(count) * 2

    dup[even] = firstdup
    dup[even+1] = firstdup + 1

    ind = ind[dup]
    vec = vec[dup]
    subb = ind[vec]
    suba = ind[np.logical_not(vec)]
    return suba, subb


def interpol(x, y, xnew, mode="linear"):
    """
    return interpolated y-values at locations
    xnew, based on input data x, y.

    This is a substitute for the IDL 'interpol' function

    INPUTS:
        x: The x-values of the input sequence
        y: The y-values of the input sequence
        xnew: The x-values of the output sequence
        mode: (optional) one of "linear" or "spline" - the method
              of interpolation to use
    OUTPUTS:
        ynew: The y-values at each value in 'xnew'.
    """
    if mode == "spline":
        # spline = interpolate.splrep(x, y)
        # ynew = interpolate.splev(xnew, spline)
        s = interpolate.InterpolatedUnivariateSpline(x, y)
        ynew = s(xnew)
    elif mode == "linear":
        ynew = np.interp(xnew, x, y)
    elif mode == "nearest":
        s = interpolate.interp1d(x, y, kind='nearest', fill_value="extrapolate")
        ynew = s(xnew)
    else:
        raise ValueError("Unrecognised interpolation mode")

    return ynew


def t2m(t, pad=True):
    """
    converts a time series (in units of years) to
    monthly intervals

    INPUTS:
        t: The input time-series
        pad: (optional) if True, extend the series to
             a multiple of 12 months
    OUTPUTS:
        tnew: the new monthly time-series
    """
    if pad:
        t0 = math.floor(np.min(t)*12) / 12.
        t1 = math.ceil(np.max(t)*12+1) / 12.
        tnew = np.arange((t1-t0)*12) / 12. + t0
    else:
        tnew = np.floor(t*12) / 12.
    return tnew


def ts2m(x, y):
    """
    uses interpolation to up/down sample the input series
    into monthly data points (12 points per time-unit).

    e.g: an input sequence with an x-range of [0, 3)
         would output x and y sequences of 36 values,
         regardless of the number of data points in
         the input.

    INPUTS:
        x: The x (time) values of the input series
        y: The y values of the input series
    OUTPUTS:
        xnew: The monthly time-values of the output series
        ynew: The interpolated y-values at each value in 'xnew'
    """
    x0 = math.floor(np.min(x) * 12) / 12.
    x1 = math.ceil(np.max(x) * 12) / 12.
    xnew = np.arange((x1 - x0)*12, dtype=x.dtype) / 12. + x0
    ynew = interpol(x, y, xnew, mode='nearest') # FIXME: maybe undo this
    return xnew, ynew


def lag_correlate(t1, y1, t2, y2, n=13, return_lag=False):
    """computes the cross-correlation of the inputs y1(t1) and y2(t2)"""
    n = (n / 2) * 2 + 1
    c = np.arange(n, dtype=float)

    t1i = t2m(t1, pad=False)
    t2i = t2m(t2, pad=False)

    for i in range(-n/2, n/2):
        m1, m2 = match(t1i, t2i + i/12.)

        y1m = y1[m1]
        y2m = y2[m2]
        ok = np.logical_and(
            np.isfinite(y1m),
            np.isfinite(y2m)
        )

        c[i + n/2 + 1], _ = stats.pearsonr(y1m[ok], y2m[ok])
    cc = interpol(np.arange(n), c, np.arange(n*100)/100., mode="spline")
    # cc = interpol(c, np.arange(n), np.arange(n*100)/100.)
    cmax = np.max(cc)
    if return_lag:
        maxpos = np.argmax(cc)
        lag = maxpos/100. - n/2
        return cmax, lag
    return cmax


def deriv(x, y=None):
    """
    Computes the numerical differentiation of the input, using
    3-point lagrangian interpolation.

    This implementation is based on GDL's DERIV.PRO function.

    INPUTS:
        x: the x-values of the input series. If no argument
            is supplied for y, then the result is the numerical
            differential of x, assuming the data points to be
            equally spaced.
            Otherwise, the x-values are the abissca values of y.
        y: (optional) the y-values of the input series.
    OUTPUTS:
        d: The differential of the input series.
    """
    n, = x.shape
    if n < 3:
        raise ValueError("Parameters must have at least 3 points")
    if y is not None:
        if y.shape != x.shape:
            raise ValueError("Parameters must have the same size")
        x = np.asfarray(x)  # ensure x is floating-point.
        x12 = x - np.roll(x, -1)  # x1 - x2
        x01 = np.roll(x, 1) - x  # x0 - x1
        x02 = np.roll(x, 1) - np.roll(x, -1)  # x0 - x2

        # this is valid for the middle points
        d = np.roll(y, 1) * (x12 / (x01 * x02)) + \
            y * (1. / x12 - 1. / x01) - \
            np.roll(y, -1) * (x01 / (x02 * x12))

        # calculate the first & last points
        # first point
        d[0] = y[0] * (x01[1] + x02[1]) / (x01[1] * x02[1]) - \
            y[1] * x02[1] / (x01[1] * x12[1]) + \
            y[2] * x01[1] / (x02[1] * x12[1])
        # Last point
        d[-1] = -y[-3] * x12[-2] / (x01[-2] * x02[-2]) + \
            y[-2] * x02[-2] / (x01[-2] * x12[-2]) - \
            y[-1] * (x02[-2] + x12[-2]) / (x02[-2] * x12[-2])
    else:
        # equally spaced points
        d = (np.roll(x, -1) - np.roll(x, 1)) / 2.
        d[0] = (-3. * x[0] + 4. * x[1] - x[2]) / 2.
        d[-1] = (3. * x[-1] - 4. * x[-2] + x[n - 3]) / 2.
    return d


def deriv_imbie(x, y, width=0, clip=False):
    """
    returns the derivative dy/dx, optionally calculating
    the moving average of y beforehand.

    INPUTS:
        x, y: The input series
        width: (optional) the window for moving average. If
                width <= 0, no smoothing is performed
        clip: (optional) The clip argument to pass the move_av
    OUTPUTS:
        dy/dx: the derivative of the input data.
    """
    if width:
        ys = move_av(width, x, y, clip=clip)
    else:
        ys = y
    return deriv(x, ys)


def move_av(dx, x, y=None, clip=False):
    """
    calculate the moving average of a function.

    INPUTS:
        dx	: the x-distance over which to average
        x   : the x-coordinates of the input data
        y   : the y-coordinates of the input data
        clip: limit the output data to values for which
              enough data exists to form a complete average.
    OUTPUT:
        ry  : the averaged y-values.

    if an argument for y is not provided, the function
    instead provides a moving average of the signal x,
    over a width of dx samples.
    """
    if y is not None:
        n = len(x)

        ry = np.empty(n,)
        ry.fill(np.NAN)

        for i in range(n):
            ok = np.logical_and(
                x > x[i] - dx/2.,
                x < x[i] + dx/2.)
            if ok.any():
                ry[i] = np.mean(y[ok])
        if clip:
            ok = np.logical_or(
                x < np.min(x) + dx/2,
                x > np.max(x) - dx/2)
            if ok.any():
                ry[ok] = np.NAN
        return ry
    else:
        ret = np.cumsum(x, dtype=x.dtype)
        ret[dx:] = ret[dx:] - ret[:-dx]
        return ret[dx - 1:] / dx


def smooth_imbie(x, y, width=43):
    """essentially an alias for move_av, with a default window width."""
    return move_av(width, x, y)


def rmsd(a, b):
    """
    returns the root-mean-squared error between the sequences 'a' and 'b'.

    INPUTS:
        a: The first series to compare
        b: The second series to compare
    OUTPUTS:
        the RMS error between 'a' and 'b'
    """
    return np.mean(np.power(a-b, 2)) ** .5


def get_offset(t1, y1, t2, y2):
    """returns the offset between two data series"""
    ok = np.logical_and(
        t1 > np.min(t2),
        t1 < np.max(t2)
    )
    if ok.any():
        y21 = interpol(t2, y2, t1)
        offset = np.nanmean(y21[ok] - y1[ok])
    else:
        offset = 0
    return offset


def annual_av(t, y, pad=False, verbose=False, spline=False):
    """returns the average per-year value of the input sequence"""
    interp_mode = "spline" if spline else "linear"
    if pad:
        t1 = t2m(t)
        y1 = interpol(y, t, t1)
    else:
        t1 = t
        y1 = y

    ny = int(math.ceil(np.max(t1)) - math.floor(np.min(t1)))
    t2 = np.arange(ny, dtype=t1.dtype) + math.floor(np.min(t1)) + .5
    y2 = np.zeros([ny], dtype=y1.dtype)

    for i in range(ny):
        ok = (np.floor(t1) == math.floor(t2[i]))
        if ok.any():
            y2[i] = np.mean(y1[ok])
    y3 = interpol(y2, y2, t1, mode=interp_mode)
    if verbose:
        # do plots
        pass
    return t1, y3


def ts_combine(t, y, nsigma=0, error=False, average=False, verbose=False, ret_data_out=False):
    """
    Combines a number of input sequences

    INPUTS:
        t: an iterable of the time-series arrays to be combined
        y: an iterable of the y-value arrays to be combined
        nsigma: (optional) tolerance within which to consider a value to be valid
        average: (optional) if True, performs a moving average on the output
        verbose: (optional) if True, renders graphs for debugging purposes
        ret_data_out (optional): if True, returns the data_out array.
    OUTPUTS:
        t1: The abissca series of the combined data
        y1: The y-values of the combined data
        data_out (optional): returns the full data set
    """
    colors = ['r', 'g', 'b', 'c', 'y', 'm', 'o', 'k']
    if verbose:
        for ti, yi, c in zip(t, y, colors[1:]):
            plt.plot(ti, yi, c+'-')
    # create _id array, which indicates which input array each element originated from
    _id = [np.ones(ti.shape, dtype=int)*(i+1) for i, ti in enumerate(t)]
    _id = np.concatenate(_id)
    # chain together input sequences
    t = np.concatenate(t)
    y = np.concatenate(y)

    # sort the input time-values, and interpolate them to monthly values
    t1 = t2m(np.sort(t))
    # remove duplicates from where inputs have overlapped
    t1 = np.unique(t1)

    # create output array
    y1 = np.zeros(t1.shape, dtype=t1.dtype)
    # c1 is used to count the number of input data points that have been used for each output point
    c1 = np.zeros(t1.shape, dtype=t1.dtype)

    # create data_out array
    data_out = np.empty(
        (len(t1), np.max(_id) + 1),
        dtype=t1.dtype
    )
    # init. all values to NaN
    data_out.fill(np.NAN)

    data_out[:, 0] = t1
    for i in range(1, np.max(_id) + 1):
        # find valid data-points where the id matches the current input seq. being worked on
        ok = np.logical_and(
            _id == i, np.isfinite(y)
        )
        if nsigma:
            # if nsigma has been specified, eliminate values far from the mean
            ok[ok] = np.abs(y[ok] - np.nanmean(y)) < max(nsigma, 1)*np.nanstd(y)
        # if we've eliminated all values in the current input, skip to the next one.
        if not ok.any():
            continue
        # get the valid items
        ti = t[ok]
        yi = y[ok]
        # sort by time
        o = np.argsort(ti)
        ti = ti[o]
        yi = yi[o]

        # match time to monthly values
        t2 = t2m(ti)
        # use interpolation to find y-values at the new times
        y2 = interpol(ti, yi, t2)
        # find locations where the times match other items in the input
        m1, m2 = match(np.floor(t1 * 12), np.floor(t2 * 12))
        # match,fix(t1*12),fix(t2*12),m1,m2
        # print repr(y1), repr(y2), repr(m1), repr(m2)
        if verbose:
            plt.plot(t2, y2, colors[i]+'.')
        # add the values from the current input seq. to the output seq.
        if error:
            y1[m1] += y2[m2] ** 2.
        else:
            y1[m1] += y2[m2]
        data_out[m1, i] = y2[m2]
        # increment the values in c1 for each current point
        c1[m1] += 1
    # set any zeros in c1 to ones
    c11 = np.maximum(c1, np.ones(c1.shape))
    # use c1 to calculate the element-wise average of the data
    if error:
        y1 = np.sqrt(y1 / c11) / np.sqrt(c11)
    else:
        y1 /= c11

    # find any locations where no values were found
    ok = (c1 == 0)
    # set those locations to NaNs
    if ok.any():
        y1[ok] = np.NAN
    # optionally plot output
    if verbose:
        plt.plot(t1, y1, '--k')
    # optionally perform moving average
    if average:
        y1 = move_av(13./12, t1, y1)
        if verbose:
            plt.plot(t1, y1, '---k')
    data_out = data_out.T
    # render the plot
    if verbose:
        plt.show()
    # return the outputs
    if ret_data_out:
        return t1, y1, data_out
    return t1, y1


def fit_imbie(x, y, fit=1, x_range=None, sigma=None, width=0, full=False):
    """
    Performs a 1st order (straight-line) polynomial fit on the input data x, y.

    The original IDL code contains 5th & 6th methods, using IDL's mpfitexpr function.
    The does not seem to be an equivalent for this in scipy, and the 5th & 6th modes
    are not used in the original code, so these methods have not been implemented.

    INPUTS:
        x: the x-values of the data set
        y: the y-values of the data set
        fit: the fitting mode to use [1...6]
        x_range: (optional) the minimum and maximum x-values to consider
        sigma: (optional) an array of weights by which to adjust the fit
        width: (modes 2 & 3) the width of the moving average
        full: (optional) if true, return y-values of the fit, and the errors
    RETURN:
        ifit: the coefficients of the polynomial fit
        yfit: (optional) the y-values of the fit line at each x-value
        yerr: (optional) the error between yfit and y
    """
    if sigma is not None:
        weights = np.array(1./sigma)
    else:
        weights = None
    if x_range is None:
        x_range = np.min(x), np.max(x)

    def func_fit5(p):
        return p[0] + p[1]*xok + p[2]*np.sin(2*np.pi*xok + p[3])

    def func_fit6(p):
        return p[0] + p[1]*np.sin(2*np.pi*xok + p[2])

    ok = np.logical_and(x >= x_range[0], x <= x_range[1])
    ifit = np.array([])

    xok = x[ok]
    yok = y[ok]

    if np.count_nonzero(ok) > 2:
        if fit == 1:
            ifit = np.polyfit(xok, yok, 1, w=weights)
        elif fit == 2:
            ifit = np.polyfit(xok, move_av(width*12, yok), 1, w=weights)
        elif fit == 3:
            ifit = np.polyfit(xok, move_av(width, xok, yok), 1, w=weights)
        elif fit == 4:
            k = np.ones([width*12])
            ifit = np.polyfit(xok, np.convolve(np.convolve(yok, k), k), 1, w=weights)
        elif fit == 5:
            sol = optimize.root(lambda p: func_fit5(p) - yok, [.05, .0, .1, .0], method='lm')
            ifit = sol.x
        elif fit == 6:
            sol = optimize.root(lambda p: func_fit6(p) - yok, [.05, .1, .0], method='lm')
            ifit = sol.x
        else:
            raise ValueError("Fit mode {} is not a valid option".format(fit))
    if full:
        if fit == 5:
            yfit = func_fit5(ifit)
            yerr = np.abs(yfit - y)
        elif fit == 6:
            yfit = func_fit6(ifit)
            yerr = np.abs(yfit - y)
        else:
            yfit = np.poly1d(ifit)(x)
            yerr = np.abs(yfit - y)
        return ifit, yfit, yerr
    return ifit


def fit_imbie2(t1, m1, t2, m2, t3, dmdt3, t4, dmdt4, verbose=False):
    """
    IMBIE Fitting method 2
    """
    tmin = min(
        np.min(t1), np.min(t2),
        np.min(t3), np.min(t4)
    )
    tmin = np.floor(tmin / 10) * 10.

    Sx1y1 = np.sum((t1 - tmin) * m1)
    Sx2y2 = np.sum((t2 - tmin) * m2)
    Sy1 = np.sum(m1)
    Sy2 = np.sum(m2)
    Sy3 = np.sum(dmdt3)
    Sy4 = np.sum(dmdt4)
    Sx12 = np.sum((t1 - tmin) * (t1 - tmin))
    Sx22 = np.sum((t2 - tmin) * (t2 - tmin))
    Sx1 = np.sum(t1 - tmin)
    Sx2 = np.sum(t2 - tmin)

    m = 2. * (Sx1y1 + Sx2y2 + Sy3 + Sy4 - 2. * Sy1 - 2. * Sy2) /\
             (Sx12 + Sx22 + 2. + 4. * Sx1 + 4. * Sx2)
    if verbose:
        plt.plot(t1, m1, t3, np.cumsum(dmdt3),
                 t2, m2, t4, np.cumsum(dmdt4))
        plt.show()
    return m


def fit_imbie3(t1, m1, t2, m2, t3, m3, t4, m4, verbose=False, full=False):
    """
    IMBIE Fitting method 3
    """
    tmin = min(
        np.min(t1), np.min(t2),
        np.min(t3), np.min(t4)
    )
    tmin = np.floor(tmin / 10) * 10.

    n1 = len(t1)
    n2 = len(t2)
    n3 = len(t3)
    n4 = len(t4)
    Sy1 = np.sum(m1)
    Sy2 = np.sum(m2)
    Sy3 = np.sum(m3)
    Sy4 = np.sum(m4)
    Sx1 = np.sum(t1 - tmin)
    Sx2 = np.sum(t2 - tmin)
    Sx3 = np.sum(t3 - tmin)
    Sx4 = np.sum(t4 - tmin)
    Sx1y1 = np.sum((t1 - tmin) * m1)
    Sx2y2 = np.sum((t2 - tmin) * m2)
    Sx3y3 = np.sum((t3 - tmin) * m3)
    Sx4y4 = np.sum((t4 - tmin) * m4)
    Sx12 = np.sum((t1 - tmin) * (t1 - tmin))
    Sx22 = np.sum((t2 - tmin) * (t2 - tmin))
    Sx32 = np.sum((t3 - tmin) * (t3 - tmin))
    Sx42 = np.sum((t4 - tmin) * (t4 - tmin))

    m = ((Sx1y1 - Sx1 * Sy1) / n1 * n1 + (Sx2y2 - Sx2 * Sy2) / n2 * n2 +
         (Sx3y3 - Sx3 * Sy3) / n3 * n3 + (Sx4y4 - Sx4 * Sy4) / n4 * n4) /\
        ((Sx12  - Sx1 * Sx1) / n1 * n1 + (Sx22  - Sx2 * Sx2) / n2 * n2 +
         (Sx32  - Sx3 * Sx3) / n3 * n3 + (Sx42  - Sx4 * Sx4) / n4 * n4)
    c1 = (Sy1 - m * Sx1) / n1
    c2 = (Sy2 - m * Sx2) / n2
    c3 = (Sy3 - m * Sx3) / n3
    c4 = (Sy4 - m * Sx4) / n4

    if verbose:
        plt.plot(
            t1, m1 - c1,
            t2, m2 - c2,
            t3, m3 - c3,
            t4, m4 - c4
        )
        plt.show()
    if full:
        return m, c1, c2, c3, c4
    return m

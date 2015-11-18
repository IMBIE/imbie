import unittest
import numpy as np

import imbie.functions as funcs


class TestFunctions(unittest.TestCase):
    """tests for functions in imbie.functions"""

    def test_match(self):
        a = np.array([3, 5, 7, 9, 11])
        b = np.array([5, 6, 7, 8, 9, 10])

        ai, bi = funcs.match(a, b)

        check = np.alltrue(a[ai] == b[bi])
        self.assertTrue(check)

    def test_interpol_linear(self):
        x = np.array([0, 1])
        y = np.array([0, 10])
        xnew = np.array([0., .5, 1.])

        ynew = funcs.interpol(x, y, xnew)

        check = np.alltrue(ynew == np.array([0, 5, 10]))
        self.assertTrue(check)

    def test_t2m(self):
        time = np.array([0, 1])

        months = funcs.t2m(time)

        self.assertTrue(
            np.allclose(months, np.linspace(0, 1, 13))
        )

    def test_ts2m(self):
        t = np.array([0, 1])
        y = np.array([0, 10])

        exp_t = np.linspace(0, 1, 13)[:-1]
        exp_y = np.linspace(0, 10, 13)[:-1]

        t_out, y_out = funcs.ts2m(t, y)

        self.assertTrue(np.allclose(t_out, exp_t))
        self.assertTrue(np.allclose(y_out, exp_y))

    def test_deriv(self):
        d = funcs.deriv(
            np.arange(20),
            np.arange(20)
        )
        self.assertTrue(np.allclose(d, np.ones([20])))

    def test_move_av(self):
        x = np.linspace(0, np.pi*16, 600)
        y = np.sin(x)

        y_av = funcs.move_av(np.pi*2, x, y, clip=True)
        fin = np.isfinite(y_av)

        self.assertTrue(
            np.allclose(0, y_av[fin], atol=1e-2)
        )

    def test_rmsd(self):
        out = funcs.rmsd(
            np.arange(10),
            np.arange(10) + 5
        )
        self.assertAlmostEqual(out, 5.)

    def test_get_offset(self):
        t = np.arange(10)
        y1 = np.linspace(0, 5, 10)
        y2 = np.linspace(1, 6, 10)

        out = funcs.get_offset(t, y1, t, y2)
        self.assertAlmostEqual(out, 1.)

    def test_annual_av(self):
        t = np.arange(10) / 12.
        y = np.linspace(0, 5, 10)

        t2, y2 = funcs.annual_av(t, y)

        self.assertTrue(np.allclose(t, t2))
        self.assertTrue(np.allclose(y2, 2.5))

    def test_ts_combine(self):
        t1 = np.linspace(0, 10, 30)
        t2 = np.linspace(5, 15, 20)

        y1 = np.ones([30])
        y2 = np.ones([20]) * 2.

        t, y = funcs.ts_combine([t1, t2], [y1, y2])

        self.assertTrue(np.allclose(t, np.linspace(0, 15, 15*12+1)))
        ok = np.isfinite(y)
        self.assertTrue(np.allclose(y[ok], 1.))

    def test_fit_imbie(self):
        x = np.linspace(0, 10, 10)
        y = np.linspace(3, 5, 10)

        params, poly, err = funcs.fit_imbie(x, y, full=True)

        self.assertAlmostEqual(params[0], .2)
        self.assertAlmostEqual(params[1], 3.)
        self.assertTrue(np.allclose(0., err))


if __name__ == "__main__":
    unittest.main()

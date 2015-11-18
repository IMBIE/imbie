import matplotlib as mpl

__doc__ = """
    This module provides a function for overriding the
    default matplotlib chart style with a custom configuration.
"""

style = {
    "font.size": 12.,

    "figure.facecolor": 'white',
    "figure.edgecolor": '0.5',

    "patch.linewidth": 0.,
    "patch.antialiased": True,

    # "legend.framealpha": 0,
    "legend.frameon": False,

    "axes.axisbelow": True,
    "axes.titlesize": "x-large",
    "axes.labelsize": "large",

    "xtick.major.width": 1.,
    "xtick.major.size": 12.,
    "xtick.minor.size": 6.,

    "ytick.major.width": 1.,
    "ytick.major.size": 12.,
    "ytick.minor.size": 6.,

    # "errorbar.capsize": 5
    "lines.linewidth": 1,
    "lines.antialiased": True
}


def apply_style(name=None, override=False, **config):
    style.update(config)
    if name is not None:
        mpl.style.use(name)
    if name is None or override:
        mpl.rcParams.update(style)

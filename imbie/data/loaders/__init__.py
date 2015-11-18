from .grace import GraceLoader
from .icesat import ICESatLoader
from .ra import RALoader
from .racmo import RacmoLoader
from .rignot import RignotLoader
from .dm import DMLoader

__doc__ = """
This module contains a number of Loader classes.

Each of these classes is intended to load & parse an input file. Instances
of these classes are used by a DataLoader instance to load and amalgamate the
input data to be processed.
"""
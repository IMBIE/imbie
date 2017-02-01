from .groups import BasinGroup
from .zwally import ZwallyBasin
from .rignot import RignotBasin
from .sheets import IceSheet

from typing import Union

Basin = Union[ZwallyBasin, RignotBasin, IceSheet]
from imbie2.const.groups import Group
from typing import Type, Dict, Any, Sequence
import matplotlib.pyplot as plt
import numpy as np


def color_variant(hex_color, brightness_offset=1):
    """
    takes a color like #87c95f and produces a lighter or darker variant
    from :https://chase-seibert.github.io/blog/2011/07/29/python-calculate-lighterdarker-rgb-colors.html
    """

    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"
    return "#" + "".join([hex(i)[2:] for i in new_rgb_int])


class ColorCollection:
    def __init__(self, valdict: Dict[Any, str]=None, keytype: Type=str, **values: str):
        if valdict is None:
            self._vals = {}
        else:
            self._vals = valdict
        self._vals.update(values)
        self.keytype = keytype

    def __getitem__(self, key: str) -> str:
        if self.keytype is not None:
            if not isinstance(key, self.keytype):
                try:
                    key = self.keytype(key)
                except ValueError:
                    raise KeyError(key)
        return self._vals[key]


class UsersColorCollection(ColorCollection):
    def __init__(self, userlist: Sequence[str], colormap=None):
        if colormap is None:
            colormap = plt.cm.nipy_spectral
        indicies = np.linspace(0, 1, len(userlist))

        keys = {u: colormap(i) for i, u in zip(indicies, userlist)}
        super().__init__(keys)


primary = ColorCollection(
    {Group.ra:  "#a03764", # "#a50f15",# "#fb6a4a", # "#ff0000",
     Group.gmb: "#5f820a", # "#82d78c",# "#a03764", # "#00ff00",
     Group.iom: "#08519c", # "#3182bd",# "#6baed6", # "#0000ff",
     Group.gia: "#ffff00",
     Group.smb: "#00ffff",
     Group.la: "#ff00ff",
     Group.all: "#888888"},
    keytype=Group
)
secondary = ColorCollection(
    {Group.ra:  "#d782a0", # "#de2d26",# "#fbb3a3", # "#ff8888",
     Group.gmb: "#64aa2d", # "#befad7",# "#a06880", # "#88ff88",
     Group.iom: "#3182bd", # "#6baed6",# "#a1c2d6", # "#8888ff",
     Group.gia: "#ffff88",
     Group.smb: "#88ffff",
     Group.la: "#ff88ff",
     Group.all: "#cccccc"},
    keytype=Group
)

from imbie2.const.groups import Group


__all__ = ["primary", "secondary"]

class ColorCollection:
    def __init__(self, valdict=None, keytype=str, **values):
        if valdict is None:
            self._vals = {}
        else:
            self._vals = valdict
        self._vals.update(values)
        self.keytype = keytype

    def __getitem__(self, key):
        if self.keytype is not None:
            if not isinstance(key, self.keytype):
                try:
                    key = self.keytype(key)
                except ValueError:
                    raise KeyError(key)
        return self._vals[key]


primary = ColorCollection(
    {Group.ra:  "#ff0000",
     Group.gmb: "#00ff00",
     Group.iom: "#0000ff",
     Group.gia: "#ffff00",
     Group.smb: "#00ffff",
     Group.all: "#444444"},
    keytype=Group
)
secondary = ColorCollection(
    {Group.ra:  "#ff8888",
     Group.gmb: "#88ff88",
     Group.iom: "#8888ff",
     Group.gia: "#ffff88",
     Group.smb: "#88ffff",
     Group.all: "#888888"},
    keytype=Group
)

class FileParserError(Exception):
    def __init__(self, instance, message):
        self.instance = instance
        self.message = message

    def __repr__(self):
        return "Error while reading {}: \"{}\"".format(
            self.instance, self.message
        )

class ParsingError(Exception):
    def __init__(self, instance, message, position=None, fragment=None):
        self.instance = instance
        self.message = message
        self.position = position
        self.fragment = fragment

    def __repr__(self):
        msg = "Error while parsing {}".format(self.instance)
        if self.position is not None:
            msg += " at line {}".format(self.position)

        msg += ":\n\t\"{}\"".format(self.message)

        if self.fragment is not None:
            msg += "\n\tin : {}".format(self.fragment)

        return msg


class ParsingWarning(Warning):
    def __init__(self, instance, message, position=None, fragment=None):
        self.instance = instance
        self.message = message
        self.position = position
        self.fragment = fragment

    def __repr__(self):
        msg = "Error while parsing {}".format(self.instance)
        if self.position is not None:
            msg += " at line {}".format(self.position)

        msg += ":\n\t\"{}\"".format(self.message)

        if self.fragment is not None:
            msg += "\n\tin : {}".format(self.fragment)

        return msg

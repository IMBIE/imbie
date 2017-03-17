from typing import Any, Sequence

from .config_errors import ParameterTypeError, ParameterValueError, MissingParameterError


class ConfigParam:
    def __init__(self, name: str, type: type, optional: bool=False, default: Any=None,
                 options: Sequence[Any]=None, multiple: bool=False):

        self.name = name
        self.type = type

        if default is not None:
            self.default = default
            self.optional = True
        else:
            self.default = None
            self.optional = optional

        self.multi = multiple
        self.options = options

    def __get__(self, instance: "ConfigFile", instance_type: type=None):

        if instance is None:
            return None

        try:
            value = instance._get_value(self.name)

        except MissingParameterError:
            if self.optional:
                return self.default
            elif self.multi:
                return []
            else:
                raise

        else:
            if self.multi:
                value = [self._cast_value(v) for v in value]
            else:
                value = self._cast_value(value[0])

        return value

    def _cast_value(self, value):
        if self.type is not None:
            try:
                value = self.type(value)
            except ValueError:
                raise ParameterTypeError(self.name, self.type, type(value))
        if self.options is not None and value not in self.options:
            raise ParameterValueError(self.name, self.options, value)

        return value


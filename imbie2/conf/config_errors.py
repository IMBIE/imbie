from typing import Sequence, Any


class ParameterTypeError(Exception):
    def __init__(self, param_name: str, param_type: type, actual_type: type):
        message = "parameter \"{}\" expected type {}, got type {}".format(
            param_name, param_type, actual_type
        )
        super().__init__(message)

class ParameterValueError(Exception):
    def __init__(self, param_name: str, param_values: Sequence[Any], actual_value: Any):
        options = ", ".join(str(i) for i in param_values)

        message = "parameter \"{}\" expected value in [{}], got value {}".format(
            param_name, options, actual_value
        )
        super().__init__(message)

class MissingParameterError(Exception):
    def __init__(self, param_name: str):
        message = "parameter \"{}\" has not been defined".format(param_name)
        super().__init__(message)

class UnknownParameterWarning(Warning):
    def __init__(self, param_name: str):
        message = "unexpected parameter in file: \"{}\"".format(param_name)
        super().__init__(message)
import enum


class AverageMethod(enum.Enum):
    equal_groups = "eqg"
    equal_series = "eqs"
    inverse_errs = "inv"
    split_altimetry = "qrt"

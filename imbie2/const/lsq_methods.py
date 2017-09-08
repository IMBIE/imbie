from enum import Enum


class LSQMethod(Enum):
    regress = 'matlab_regress'
    normal = 'ordinary_least_squares'
    weighted = 'weighted_least_squares'

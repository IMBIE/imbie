from enum import Enum


class ErrorMethod(Enum):
    rms = "rms"  # root mean squared
    rss = "rss"  # root sum squared
    sum = "sum"
from enum import Enum


class ErrorMethod(Enum):
    rms = "root mean squared"
    rss = "root sum squared"
    sum = "sum"
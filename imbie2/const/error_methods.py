from enum import Enum


class ErrorMethod(Enum):
    rms = "rms"  # root mean squared
    rss = "rss"  # root sum squared
    sum = "sum"
    imbie1 = "imbie1" # RMS / sqrt(N)
    average = "avg"
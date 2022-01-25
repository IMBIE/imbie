from enum import Enum


class ErrorMethod(Enum):
    rms = "rms"  # root mean squared
    rss = "rss"  # root sum squared
    sum = "sum"
    imbie1 = "imbie1" # RMS / sqrt(N)
    average = "avg"
    rms_deviation = "dev" # rms deviation from average series
    constant_dev = "dev_const"
    rss_dev_epoch = "dev_rss_epoch"
    max_error = "max"

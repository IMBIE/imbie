from enum import IntEnum


class ErrorCode(IntEnum):
    """
    return codes for errors
    """
    logging = 1  # cannot open log file
    input_path = 2  # input path does not exist
    no_data = 3  # no data in input path
    config_missing = 4  # config does not exist
    config_invalid = 5  # config does not parse
    output_path = 6  # cannot write to output path

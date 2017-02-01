#! /usr/bin/python3
import csv
from abc import ABCMeta, abstractmethod
import enum
import warnings
import numpy as np
from collections import OrderedDict
import logging

from imbie2.model.series import MassChangeDataSeries, MassRateDataSeries
from imbie2.const.basins import *
from .errors import *


class FileParser(metaclass=ABCMeta):
    """
    class for reading standard format-compliant IMBIE data
    files. Returns the data as a collection of DataSeries
    instances.

    something like this:

    >>> basins = {}
    >>> with FileParser('example.csv') as fp:
    ...   for data_series in fp:
    ...     basins[data_series.basin_id] = data_series
    """

    def __init__(self, filename, user_group, logger=None, user_name=None):
        self.filename = filename
        self._file = None
        self._csv = None
        self._data = None
        self.user_group = user_group
        self.user_name = user_name

        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger


    def open(self, filename=None):
        """
        opens the specified file (defaults to the
        file specified in the constructor)
        """
        if filename is not None:
            self.filename = filename
        self._file = open(self.filename, 'r', newline='')
        self._csv = csv.reader(self._file)

        self.parse_file()

    def close(self):
        """
        closes the file
        """
        self._csv = None
        self._file.close()

    def __enter__(self):
        """
        called at start of 'with' statement. Opens the file
        and returns the parser instance.
        """
        try:
            self.open()
            return self
        except:  # (FileParserError, ParsingError):
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        called at end of 'with' statement. Closes the file.
        """
        self.close()

    def __repr__(self):
        return "FileParser({})".format(self.filename)

    def __iter__(self):
        if self._data is None:
            raise FileParserError(self, "file has not been opened")
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def parse_basin(self, basin_grp, basin_id, lnum=None):
        b = basin_id.split(':')[-1].strip()
        try:
            grp = BasinGroup(basin_grp.lower())
        except ValueError:
            grp = None

        if IceSheet.is_valid(basin_id.lower()):
            _id = IceSheet.get_basin(basin_id.lower())
            if grp is None: grp = BasinGroup.sheets
        elif grp == BasinGroup.zwally:
            _id = ZwallyBasin.parse(basin_id)
        elif grp == BasinGroup.rignot:
            _id = RignotBasin.parse(basin_id)
        else:
            e = "{}: invalid basin on line {} ({}: {})".format(
                self.filename, lnum, basin_grp, basin_id)
            self.logger.warn(e)
            # warnings.warn(
            #     ParsingWarning(self, "invalid basin definition", lnum, e)
            # )
            return None, None
        return grp, _id

    def read_lines(self, columns=9):
        for lnum, line in enumerate(self._csv):
            if not line: # empty line
                continue
            if line[0].lower() == 'experiment group': # header line
                self.logger.debug(line)
                continue
            if line[0][0] == '#': # comment
                continue

            # check that this line is long enough
            line = [item.strip() for item in line if item]

            if len(line) < columns:
                msg = "{}: missing columns (expected {}, got {})".format(
                    self.filename, columns, len(line)
                )
                line = '\n' + ", ".join(line)
                self.logger.error(msg + line)
                continue

            yield lnum, line

    @abstractmethod
    def parse_file(self):
        pass


class MassChangeParser(FileParser):
    """
    child of the FileParser class for reading change-of-mass data files.
    This will create a collection of MassChangeDataSeries instances, one
    for each drainage basin defined in the file.
    """
    class WorkingCollection:
        """
        instances of this class are used to store data while the file is being
        parsed. Each one accumulates data points relating to a single drainage
        basin. Once every item in the file has been read, the build_series method
        is used to create a MassChangeDataSeries instance of the collection's
        sorted data.
        """
        def __init__(self, name, user_group, data_group,
                           basin_grp, basin_id, basin_area):
            self.basin_grp = basin_grp
            self.basin_id = basin_id
            self.basin_area = basin_area

            self.user_name = name
            self.user_group = user_group
            self.data_group = data_group

            self.time = []
            self.mass = []
            self.area = []
            self.errs = []

        def add_item(self, time, mass, area, err):
            """
            add a new data point to the collection
            """
            self.time.append(time)
            self.mass.append(mass)
            self.area.append(area)
            self.errs.append(err)

        def build_series(self):
            """
            sort the data and return a MassChangeDataSeries instance
            """
            time = np.asarray(self.time, dtype=np.float64)
            mass = np.asarray(self.mass, dtype=np.float64)
            area = np.asarray(self.area, dtype=np.float64)
            errs = np.asarray(self.errs, dtype=np.float64)

            # get indicies to sort arrays by time
            i = np.argsort(time)

            return MassChangeDataSeries(
                self.user_name, self.user_group,
                self.data_group,
                basin_group=self.basin_grp,
                basin_id=self.basin_id,
                basin_area=self.basin_area,
                time=time[i],
                area=area[i],
                mass=mass[i],
                errs=errs[i]
            )

    def parse_file(self):
        """
        reads the data from the file
        """
        # empty the data dictionary
        self._data = []
        collections = OrderedDict()

        for lnum, line in self.read_lines():
            # expected:
            #  0 :surname, 1: experiment group, 2: basin group, 3: basin id,
            #  ... 4: basin area, 5: observed area, 6: date, 7: dM, 8: dM uncertainty
            try:
                name, data_group = line[0], line[1]
                basin_grp, basin_id = self.parse_basin(line[2], line[3], lnum)

                if basin_id is None: continue

                basin_area = float(line[4])
                obs_area = float(line[5])
                time = float(line[6])
                mass = float(line[7])
                err = float(line[8])

            except ValueError as e:
                self.logger.warn(e)
                # warnings.warn(ParsingWarning(self, "error parsing value", lnum, e))
                continue

            if (basin_id, basin_grp) in collections:
                col = collections[(basin_id, basin_grp)]
            else:
                # create a new collection for this basin
                if self.user_name is not None:
                    name = self.user_name

                col = self.WorkingCollection(
                    name, self.user_group, data_group,
                    basin_grp, basin_id, basin_area
                )
                collections[(basin_id, basin_grp)] = col

            # add the data point to the appropriate collection
            col.add_item(time, mass, obs_area, err)

        # create MassChangeDataSeries instances for every WorkingCollection,
        #  and store the results in _data
        self._data = [
            col.build_series() for col in collections.values()
        ]


class MassRateParser(FileParser):
    """
    child of the FileParser class for reading change-of-mass data files.
    This will create a collection of MassChangeDataSeries instances, one
    for each drainage basin defined in the file.
    """
    class WorkingCollection:
        """
        instances of this class are used to store data while the file is being
        parsed. Each one accumulates data points relating to a single drainage
        basin. Once every item in the file has been read, the build_series method
        is used to create a MassRateDataSeries instance of the collection's
        sorted data.
        """
        def __init__(self, name, user_group, data_group, basin_grp, basin_id,
                     basin_area):
            self.basin_grp = basin_grp
            self.basin_id = basin_id
            self.basin_area = basin_area

            self.user_name = name
            self.user_group = user_group
            self.data_group = data_group

            self.time_0 = []
            self.time_1 = []
            self.rate = []
            self.area = []
            self.errs = []

        def add_item(self, time_0, time_1, rate, area, err):
            """
            add a new data point to the collection
            """
            self.time_0.append(time_0)
            self.time_1.append(time_1)
            self.rate.append(rate)
            self.area.append(area)
            self.errs.append(err)

        def build_series(self):
            """
            sort the data and return a MassChangeDataSeries instance
            """
            time_0 = np.asarray(self.time_0, dtype=np.float64)
            time_1 = np.asarray(self.time_1, dtype=np.float64)
            rate = np.asarray(self.rate, dtype=np.float64)
            area = np.asarray(self.area, dtype=np.float64)
            errs = np.asarray(self.errs, dtype=np.float64)

            # get indicies to sort arrays by time
            i = np.argsort(time_0)

            return MassRateDataSeries(
                self.user_name, self.user_group,
                self.data_group,
                basin_group=self.basin_grp,
                basin_id=self.basin_id,
                basin_area=self.basin_area,
                t_start=time_0[i],
                t_end=time_1[i],
                area=area[i],
                rate=rate[i],
                errs=errs[i]
            )

    def parse_file(self):
        """
        reads the data from the file
        """
        # empty the data dictionary
        self._data = []
        collections = OrderedDict()

        for lnum, line in self.read_lines(columns=10):
            # expected:
            #  0 :surname, 1: experiment group, 2: basin group, 3: basin id,
            #  ... 4: basin area, 5: observed area, 6: start date, 7: end data,
            #  ... 8: dM/dt, 9: dM/dt uncertainty
            try:
                name, data_group = line[0], line[1]
                basin_grp, basin_id = self.parse_basin(line[2], line[3], lnum)

                if basin_grp is None: continue

                basin_area = float(line[4])
                obs_area = float(line[5])
                time0 = float(line[6])
                time1 = float(line[7])
                rate = float(line[8])
                err = float(line[9])

            except ValueError as e:
                self.logger.warn(e)
                # warnings.warn(ParsingWarning(self, "error parsing value", lnum, e))
                continue

            if (basin_id, basin_grp) in collections:
                col = collections[(basin_id, basin_grp)]
            else:
                if self.user_name is not None:
                    name = self.user_name
                # create a new collection for this basin
                col = self.WorkingCollection(
                    name, self.user_group, data_group, basin_grp, basin_id,
                    basin_area
                )
                collections[(basin_id, basin_grp)] = col

            # add the data point to the appropriate collection
            col.add_item(time0, time1, rate, obs_area, err)

        # create MassChangeDataSeries instances for every WorkingCollection,
        #  and store the results in _data
        self._data = [
            col.build_series() for col in collections.values()
        ]

class IOMRatesParser(MassChangeParser):
    class WorkingCollection(MassChangeParser.WorkingCollection):
        def build_series(self):
            """
            sort the data and return a MassChangeDataSeries instance
            """
            time = np.asarray(self.time, dtype=np.float64)
            rate = np.asarray(self.mass, dtype=np.float64)
            area = np.asarray(self.area, dtype=np.float64)
            errs = np.asarray(self.errs, dtype=np.float64)

            # get indicies to sort arrays by time
            i = np.argsort(time)

            return MassRateDataSeries(
                self.user_name, self.user_group,
                self.data_group,
                basin_group=self.basin_grp,
                basin_id=self.basin_id,
                basin_area=self.basin_area,
                t_start=time[i],
                t_end=time[i],
                area=area[i],
                rate=rate[i],
                errs=errs[i]
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    while True:
        try:
            fname = input()
        except EOFError:
            break

        if not fname: break
        if fname[0] == '#': continue

        print(fname)
        with MassChangeParser(fname) as data:
            for basin in data:
                plt.fill_between(
                    basin.t,
                    basin.mass-basin.errs,
                    basin.mass+basin.errs,
                    alpha=0.5
                )
                plt.plot(basin.t, basin.mass, 'b-')
                plt.grid()

                plt.title(basin.basin_id.value)
                plt.xlabel("year")
                plt.ylabel("mass change (Gt)")

                plt.show()

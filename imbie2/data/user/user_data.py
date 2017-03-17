#! /usr/bin/python3
from imbie2.data.csv import MassChangeParser, MassRateParser, IOMRatesParser
from imbie2.model.series import MassChangeDataSeries, MassRateDataSeries
from imbie2.model.collections import MassRateCollection, MassChangeCollection

import json
import os


class UserData:
    _data_file = '.answers.json'
    _rate_questions = {
        'GMB': 'mascons-approach-upload',
        'RA': 'mean-rate-upload',
        'IOM': 'time-series-accumulation-upload'
    }
    _mass_questions = {
        'GMB': "spherical-harmonics-upload",
        'RA': "time-series-upload",
        'IOM': "mass-balance-upload"
    }

    @property
    def rate_file(self):
        if 'files' not in self._data:
            return None
        if self.group not in self._rate_questions:
            return None

        _id = self._rate_questions[self.group]
        return self._data['files'].get(_id, None)

    @property
    def mass_file(self):
        if 'files' not in self._data:
            return None
        if self.group not in self._mass_questions:
            return None

        _id = self._mass_questions[self.group]
        return self._data['files'].get(_id, None)

    @property
    def has_rate_data(self):
        return self.rate_file is not None

    @property
    def has_mass_data(self):
        return self.mass_file is not None

    @property
    def group(self):
        return self._data.get('group', None)

    @property
    def name(self):
        return self._data.get('username', None)

    @property
    def forename(self):
        try:
            return self._data['contact']['forename']
        except KeyError: return None

    @property
    def lastname(self):
        try:
            return self._data['contact']['lastname']
        except KeyError: return None

    def __init__(self, folder):
        fpath = os.path.join(folder, self._data_file)
        self.folder = folder

        self._data = {}
        with open(fpath) as f:
            self._data.update(json.load(f))

    def _rate_series(self):
        info = self.rate_file
        path = os.path.join(self.folder, info['name'])

        if self.group == 'IOM':
            parser = IOMRatesParser
        else:
            parser = MassRateParser

        with parser(path, self.group, user_name=self.lastname) as f:
            if f is not None:
                yield from f

    def _mass_series(self):
        info = self.mass_file
        path = os.path.join(self.folder, info['name'])

        with MassChangeParser(path, self.group, user_name=self.lastname) as f:
            if f is not None:
                yield from f

    def rate_data(self, convert=False):
        if self.has_rate_data:
            yield from self._rate_series()
        elif self.has_mass_data and convert:
            for series in self._mass_series():
                yield MassRateDataSeries.derive_rates(series)

    def mass_data(self, convert=False):
        if self.has_mass_data:
            yield from self._mass_series()
        elif self.has_rate_data and convert:
            for series in self._rate_series():
                yield MassChangeDataSeries.accumulate_mass(series)

    def rate_collection(self):
        return MassRateCollection(*self.rate_data())

    def mass_collection(self):
        return MassChangeCollection(*self.mass_data())

    @classmethod
    def find(cls, root=None):
        if root is None:
            root = os.getcwd()

        for path, _, files in os.walk(root):

            if cls._data_file not in files:
                continue

            yield cls(path)

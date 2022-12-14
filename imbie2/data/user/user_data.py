#! /usr/bin/python3

#### IMBIE3 update: altered to match the new submission format, ie the files containing info about
# each submission are called blah.json, where blah is the name of the lowest-level directory
# containing them. The properties also have to be read in differently, and a username defined.
# The one-timestamp IOMRatesParser has been removed as all dm/dt files should now have the same
# format.

from imbie2.data.csv import MassChangeParser, MassRateParser, IOMRatesParser
from imbie2.model.series import MassChangeDataSeries, MassRateDataSeries
from imbie2.model.collections import MassRateCollection, MassChangeCollection

import json
import os

import unicodedata as ud


#### IMBIE3 update: make a username with no accents or special characters
# for use in the users_skip config param

def rmdiacritics(char):
    '''
    Return the base character of char, by "removing" any
    diacritics like accents or curls and strokes and the like.
    '''
    desc = ud.name(char)
    cutoff = desc.find(' WITH ')
    if cutoff != -1:
        desc = desc[:cutoff]
        try:
            char = ud.lookup(desc)
        except KeyError:
            pass  # removing "WITH ..." produced an invalid name
    return char


#### IMBIE3 update: class definitions altered to fit new submission format

class UserData:
    _data_file_ext = '.json'        
    _rate_questions = {
        'GMB': 'mass-rate-upload',
        'RA': 'mean-rate-upload',
        'LA': 'mean-rate-upload',
        'IOM': 'time-series-accumulation-upload'
    }
    _mass_questions = {
        'GMB': "time-series-upload",
        'RA': "time-series-upload",
        'LA': "time-series-upload",
        'IOM': "time-series-total-mass-change-upload"
    }

    @property
    def rate_file(self):
        if self.group not in self._rate_questions:
            return None

        _id = self._rate_questions[self.group]
        return self._data.get(_id, None)

    @property
    def mass_file(self):
        if self.group not in self._mass_questions:
            return None

        _id = self._mass_questions[self.group]
        return self._data.get(_id, None)

    @property
    def has_rate_data(self):
        return self.rate_file is not None

    @property
    def has_mass_data(self):
        return self.mass_file is not None

    @property
    def group(self):
        this_group=self._data.get('group', None)   #### IMBIE3 update:
        if 'radar' in this_group:           # search for part of string
            this_group='RA'                 # otherwise gets confused by
        elif 'gravimetry' in this_group:    # extra blanks etc
            this_group='GMB'
        elif 'mass' in this_group:
            this_group='IOM'           
        return this_group

    @property
    def name(self):    #### IMBIE3 update: username is last name in lower case, no accents
        username=self._data.get('lastname', None)
        username = username.lower()
        username=''.join(map(rmdiacritics, username))
        return username

    @property
    def forename(self):
        try:
            return self._data['forename']
        except KeyError: return None

    @property
    def lastname(self):
        try:
            return self._data['lastname']
        except KeyError: return None

    def __init__(self, folder):
        submission_dir=os.path.basename(os.path.normpath(folder))
        submission_file=submission_dir+self._data_file_ext
        fpath = os.path.join(folder, submission_file)

        self.folder = folder

        self._data = {}
        with open(fpath) as f:
            self._data.update(json.load(f))


    def _rate_series(self):
        info = self.rate_file
        path = os.path.join(self.folder, self._rate_questions[self.group], info['name'])

        parser = MassRateParser

        with parser(path, self.group, user_name=self.lastname) as f:
            if f is not None:
                yield from f

    def _mass_series(self):
        info = self.mass_file
        path = os.path.join(self.folder, self._mass_questions[self.group], info['name'])

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

    #  This method walks along the directory tree, and returns the directory
    #  containing each json file, which __init__ uses to populate the class
    
    @classmethod
    def find(cls, root=None):
        if root is None:
            root = os.getcwd()

        for path, _, files in os.walk(root):

            if not any([f.endswith('.json') for f in files]):
                continue

            yield cls(path)

import os
import scipy.io

import config


class DatasetInfo:
    pass

OMNIGLOT_PATH = os.path.join(config.DATA_DIR, 'characters', 'chardata.mat')

_characters_data = {}

class CharactersInfo(DatasetInfo):
    m = 24345
    n = 784
    num_rows = 28
    num_cols = 28

    @staticmethod
    def read_data(fname=OMNIGLOT_PATH):
        if not _characters_data:
            path = os.path.join(config.DATA_DIR, 'characters', 'chardata.mat')
            vars = scipy.io.loadmat(path)
            _characters_data['train'] = CharactersData(vars['data'].T)
            _characters_data['test'] = CharactersData(vars['testdata'].T)

    @classmethod
    def load(cls):
        cls.read_data()
        return _characters_data['train']

    @classmethod
    def load_test(cls):
        cls.read_data()
        return _characters_data['test']
    

class CharactersData(CharactersInfo):
    def __init__(self, values):
        self.values = values

    def as_matrix(self):
        return self.values.copy()


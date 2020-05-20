from __future__ import absolute_import, division, print_function, unicode_literals

from abc import abstractmethod, ABCMeta


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, feat_id, run_id, data=None):
        self.feat_id = feat_id
        self.run_id = run_id
        self.model = None

        if data is None:
            self.load()
        else:
            self.train(data)
            self.save()

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

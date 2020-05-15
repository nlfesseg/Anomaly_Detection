from __future__ import absolute_import, division, print_function, unicode_literals

from abc import abstractmethod, ABCMeta


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, feat_id, run_id, dataset=None):
        self.feat_id = feat_id
        self.run_id = run_id
        self.model = None

        if dataset is None:
            self.load()
        else:
            self.train(dataset)
            self.save()

    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

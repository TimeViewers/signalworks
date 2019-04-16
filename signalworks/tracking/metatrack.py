import abc


class MetaTrack(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_duration(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

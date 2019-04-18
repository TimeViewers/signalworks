# -*- coding: utf-8 -*-
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
from signalworks.tracking import Partition, TimeValue, Wave

Tracks = Union[Wave, TimeValue, Partition]
# type alias


class Data(NamedTuple):
    wave: Optional[Wave]
    active: Optional[Partition]


class InvalidDataError(Exception):
    pass


class InvalidParameterError(Exception):
    pass


class ProcessError(Exception):
    pass


class DefaultProgressTracker:
    def update(self, value: int) -> None:
        print(f"{value}%", end="...", flush=True)


class Processor(metaclass=ABCMeta):
    name = "Processor"
    acquire = NamedTuple("acquire", [("wave", Wave), ("active", Partition)])

    def __init__(self):
        self.data = Data(wave=None, active=None)
        self.parameters: Dict[str, Any] = {}  # default parameters
        self.progressTracker = DefaultProgressTracker()

    def set_data(self, data: Dict[str, Tracks]) -> None:
        wav = data.get("wave")
        if wav is not None and not isinstance(wav, Wave):
            raise InvalidDataError

        active = data.get("active")
        if active is not None and not isinstance(wav, Partition):
            raise InvalidDataError

        self.data = Data(wave=wav, active=active)

    def get_parameters(self) -> Dict[str, str]:
        # default parameters can be modified here based on the data
        return {k: str(v) for k, v in self.parameters.items()}

    def set_parameters(self, parameters: Dict[str, str]) -> None:
        if __debug__:
            for name, value in parameters.items():
                logging.debug(f"Received parameter {name} of value {value}")
        try:
            for key in parameters.keys():
                if type(self.parameters[key]) == np.ndarray:
                    self.parameters[key] = np.fromstring(
                        parameters[key].rstrip(")]").lstrip("[("), sep=" "
                    )
                else:
                    self.parameters[key] = type(self.parameters[key])(parameters[key])
        except Exception as e:
            raise InvalidParameterError(e)
        # additional parameter checking can be performed here

    @abstractmethod
    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[Tracks, ...]:
        pass

    def del_data(self):
        self.data = Data(wave=None, active=None)


def get_processor_classes() -> Dict[str, Callable[..., Processor]]:
    def all_subclasses(c):
        return c.__subclasses__() + [
            a for b in c.__subclasses__() for a in all_subclasses(b)
        ]

    return {obj.name: obj for obj in all_subclasses(Processor)}

from enum import StrEnum, auto


class ModelType(StrEnum):
    REGRESSOR = auto()
    HYBRID = auto()


class DataVersion(StrEnum):
    RAW = auto()
    Q4 = auto()
    FOURIER = auto()

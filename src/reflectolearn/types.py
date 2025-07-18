import enum


class ModelType(enum.Enum):
    REGRESSOR = "regressor"
    HYBRID = "hybrid"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, value: str):
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(f"Invalid model type: {value}") from e


class DataVersion(enum.Enum):
    RAW = "raw"
    Q4 = "q4"
    FOURIER = "fourier"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, value: str):
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(f"Invalid preprocessing version: {value}") from e

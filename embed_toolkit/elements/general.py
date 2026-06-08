from enum import Enum


class Laterality(Enum):
    LEFT = "L"
    RIGHT = "R"
    BILATERAL = "B"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value) -> "Laterality":
        # if the lookup failed, check if a lower-case char was input
        # otherwise return UNKNOWN
        match value:
            case "l":
                return cls.LEFT
            case "r":
                return cls.RIGHT
            case _:
                return cls.UNKNOWN


import ast
import re
from enum import Enum
from typing import Union

class ImageModality(Enum):
    FFDM = "2D"
    DBT = "3D"
    S2D = "cview"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value) -> "ImageModality":
        return cls.UNKNOWN

class ViewPosition(Enum):
    CC = "CC"
    MLO = "MLO"
    ML = "ML"
    LM = "LM"
    XCCL = "XCCL"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value) -> "ViewPosition":
        if isinstance(value, str):
            # extract view position patterns (CC, MLO, or ML) after uppercasing
            match = re.search(r"\b(CC|MLO|ML)\b", value.upper())

            # if a match was found, return the Enum member by passing the extracted string back to cls()
            if match:
                return cls(match.group(1))

        # default to UNKNOWN if value was not a string or no matches were found
        return cls.UNKNOWN


# PatientOrientation -------------------------------------------------------------------------------------------------


class OrientationDirection(Enum):
    P = "P"
    A = "A"
    L = "L"
    R = "R"
    HL = "HL"
    HR = "HR"
    FL = "FL"
    FR = "FR"
    H = "H"
    F = "F"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value) -> "OrientationDirection":
        # if the lookup failed, return UNKNOWN
        return cls.UNKNOWN


class PatientOrientation:
    def __init__(self, orientation: Union[str, tuple[str, str]]) -> None:
        # parse input
        row, col = self._parse(orientation)
        self.row: OrientationDirection = row
        self.col: OrientationDirection = col

    @property
    def exact(self) -> bool:
        # returns true if the orientation elements were matched exactly
        return (self.row != OrientationDirection.UNKNOWN) and (
            self.col != OrientationDirection.UNKNOWN
        )

    def _parse(
        self, orientation: Union[str, tuple[str, str]]
    ) -> tuple[OrientationDirection, OrientationDirection]:
        # if orientation is a str, e.g. "('P', 'R')", parse it into a tuple
        if isinstance(orientation, str):
            orientation: tuple[str, str] = ast.literal_eval(orientation)

        # if orientation is not a length 2 tuple at this stage, error
        if (not isinstance(orientation, tuple)) | (len(orientation) != 2):
            raise ValueError(f"Invalid orientation provided '{orientation}'!")

        # parse the row/col orientation strings into Direction enums
        return (
            OrientationDirection(orientation[0]),
            OrientationDirection(orientation[1]),
        )

    def __repr__(self) -> str:
        return f"PatientOrientation({self.row}, {self.col})"

    def __hash__(self) -> int:
        return hash((self.row, self.col))


# FieldOfView --------------------------------------------------------------------------------------------------------


class FovRotation(Enum):
    NONE = 0.0
    FULL = 180.0
    UNKNOWN = float("nan")

    @classmethod
    def _missing_(cls, value) -> "FovRotation":
        return cls.UNKNOWN


class FovHFlip(Enum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value) -> "FovHFlip":
        return cls.UNKNOWN

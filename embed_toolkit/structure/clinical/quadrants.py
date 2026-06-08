from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from embed_toolkit.elements.general import Laterality


class MlAxis(Enum):
    # medial/lateral axis
    LATERAL = 1.0
    CENTRAL = 0.0
    MEDIAL = -1.0
    UNKNOWN = float("nan")


class SiAxis(Enum):
    # superior/inferior axis
    SUPERIOR = 1.0
    CENTRAL = 0.0
    INFERIOR = -1.0
    UNKNOWN = float("nan")


class DepthAxis(Enum):
    ANTERIOR = 0.0
    MIDDLE = 1.0
    POSTERIOR = 2.0
    UNKNOWN = float("nan")


class LocCode(Enum):
    OU = "OU"
    L = "L"
    IN = "IN"
    D = "D"
    Z = "Z"
    X = "X"
    Y = "Y"
    W = "W"
    C = "C"
    A = "A"
    T = "T"
    UP = "UP"
    U = "U"
    LO = "LO"
    I = "I"  # noqa: E741
    S = "S"
    AN = "AN"
    MD = "MD"

    @classmethod
    def _missing_(cls, value) -> "LocCode":
        raise ValueError(f"Unrecognized Magview location code: '{value}'")

    def to_quadrant(self, side: "Laterality") -> "Quadrant":
        match self:
            case LocCode.OU | LocCode.L:
                return Quadrant(side=side, ml=MlAxis.LATERAL)
            case LocCode.IN | LocCode.D:
                return Quadrant(side=side, ml=MlAxis.MEDIAL)
            case LocCode.Z:
                return Quadrant(side=side, ml=MlAxis.MEDIAL, si=SiAxis.INFERIOR)
            case LocCode.X:
                return Quadrant(side=side, ml=MlAxis.MEDIAL, si=SiAxis.SUPERIOR)
            case LocCode.Y:
                return Quadrant(side=side, ml=MlAxis.LATERAL, si=SiAxis.INFERIOR)
            case LocCode.W:
                return Quadrant(side=side, ml=MlAxis.LATERAL, si=SiAxis.SUPERIOR)
            case LocCode.C:
                return Quadrant(side=side, ml=MlAxis.CENTRAL, si=SiAxis.CENTRAL)
            case LocCode.A | LocCode.T:
                return Quadrant(
                    side=side, si=SiAxis.SUPERIOR, depth=DepthAxis.POSTERIOR
                )
            case LocCode.UP | LocCode.U:
                return Quadrant(side=side, si=SiAxis.SUPERIOR)
            case LocCode.LO | LocCode.I:
                return Quadrant(side=side, si=SiAxis.INFERIOR)
            case LocCode.S:
                return Quadrant(
                    side=side,
                    ml=MlAxis.CENTRAL,
                    si=SiAxis.CENTRAL,
                    depth=DepthAxis.ANTERIOR,
                )
            case LocCode.AN:
                return Quadrant(side=side, depth=DepthAxis.ANTERIOR)
            case LocCode.MD:
                return Quadrant(side=side, depth=DepthAxis.MIDDLE)


class DepCode(Enum):
    A = "A"
    M = "M"
    P = "P"

    @classmethod
    def _missing_(cls, value) -> "DepCode":
        raise ValueError(f"Unrecognized Magview depth code: '{value}'")

    def to_quadrant(self, side: "Laterality") -> "Quadrant":
        match self:
            case DepCode.A:
                return Quadrant(side=side, depth=DepthAxis.ANTERIOR)
            case DepCode.M:
                return Quadrant(side=side, depth=DepthAxis.MIDDLE)
            case DepCode.P:
                return Quadrant(side=side, depth=DepthAxis.POSTERIOR)


class ClockCode(Enum):
    C1 = 1
    C2 = 2
    C3 = 3
    C4 = 4
    C5 = 5
    C6 = 6
    C7 = 7
    C8 = 8
    C9 = 9
    C10 = 10
    C11 = 11
    C12 = 12

    @classmethod
    def _missing_(cls, value) -> "ClockCode":
        raise ValueError(f"Unrecognized clock code: '{value}'")

    def to_quadrant(self, side: "Laterality") -> "Quadrant":
        match self:
            case ClockCode.C6 | ClockCode.C12:
                ml = MlAxis.CENTRAL
            case ClockCode.C1 | ClockCode.C2 | ClockCode.C3 | ClockCode.C4 | ClockCode.C5:
                ml = MlAxis.LATERAL if side is Laterality.LEFT else MlAxis.MEDIAL
            case _:  # C7–C11
                ml = MlAxis.MEDIAL if side is Laterality.LEFT else MlAxis.LATERAL

        match self:
            case ClockCode.C3 | ClockCode.C9:
                si = SiAxis.CENTRAL
            case ClockCode.C10 | ClockCode.C11 | ClockCode.C12 | ClockCode.C1 | ClockCode.C2:
                si = SiAxis.SUPERIOR
            case _:  # C4–C8
                si = SiAxis.INFERIOR

        return Quadrant(side=side, ml=ml, si=si)


def _coerce_loc_code(value: str) -> Union[LocCode, ClockCode]:
    try:
        return ClockCode(int(value))
    except ValueError:
        return LocCode(value)


@dataclass
class Quadrant:
    side: Laterality
    ml: MlAxis = MlAxis.UNKNOWN
    si: SiAxis = SiAxis.UNKNOWN
    depth: DepthAxis = DepthAxis.UNKNOWN

    @classmethod
    def from_magview_codes(
        cls,
        loc_code: Union[str, LocCode, ClockCode],
        depth_code: Optional[DepCode],
        side: Laterality,
    ) -> "Quadrant":
        # TODO: need to handle both LocCode/ClockCode here (since Magview could encode either in the loc column)
        loc_obj = _coerce_loc_code(loc_code) if isinstance(loc_code, str) else loc_code

        # build the initial quadrant with the depth code so its explicit depth takes
        # priority over any depth implied by the loc code (update() never overwrites a known axis)
        _cls: "Quadrant" = (
            depth_code.to_quadrant(side=side) if depth_code is not None else Quadrant(side=side)
        )
        _cls.update(loc_obj.to_quadrant(side=side))
        return _cls

    def update(self, target: "Quadrant") -> None:
        # this should update the quadrant with any defined (non-unknown) axis
        # definitions in the target

        # first, ensure the side of both quadrants matches
        if self.side != target.side:
            raise ValueError(f"Quadrants have mismatched lateralities! self: {self.side}, target: {target.side}")
        if self.ml == MlAxis.UNKNOWN and target.ml != MlAxis.UNKNOWN:
            self.ml = target.ml
        if self.si == SiAxis.UNKNOWN and target.si != SiAxis.UNKNOWN:
            self.si = target.si
        if self.depth == DepthAxis.UNKNOWN and target.depth != DepthAxis.UNKNOWN:
            self.depth = target.depth


# ─────────────────────────────────────────────────────────────────────────────
# RoiPosition
# ─────────────────────────────────────────────────────────────────────────────


def _quantize_si(v: float) -> SiAxis:
    if v >= 0.5:
        return SiAxis.SUPERIOR
    if v <= -0.5:
        return SiAxis.INFERIOR
    return SiAxis.CENTRAL


def _quantize_ml(v: float) -> MlAxis:
    if v >= 0.5:
        return MlAxis.LATERAL
    if v <= -0.5:
        return MlAxis.MEDIAL
    return MlAxis.CENTRAL


def _quantize_depth(v: float) -> DepthAxis:
    # thresholds at 2/3 and 4/3 match the equal-thirds partitioning used by
    # BreastLandmarkMixin._partition_depth_range(), since depth is normalized so that
    # depth_range maps to 2.0 (the DepthAxis.POSTERIOR value).
    if v < 2 / 3:
        return DepthAxis.ANTERIOR
    if v < 4 / 3:
        return DepthAxis.MIDDLE
    return DepthAxis.POSTERIOR


@dataclass
class RoiPosition:
    """Continuous breast-axis coordinates for an ROI center point.

    Values are normalized to match the scale of the corresponding axis enums:
      si    in [-1,  1]: SiAxis scale    (SUPERIOR=1.0,  CENTRAL=0.0, INFERIOR=-1.0)
      ml    in [-1,  1]: MlAxis scale    (LATERAL=1.0,   CENTRAL=0.0, MEDIAL=-1.0)
      depth in [ 0,  2]: DepthAxis scale (ANTERIOR=0.0,  MIDDLE=1.0,  POSTERIOR=2.0)

    None on any axis means that axis was not observable from the imaging view
    (e.g. ML is typically None from a single 2-D mammogram view).

    Multi-view fusion follows the same pattern as code reconciliation: convert
    each view's RoiPosition to a partial Quadrant via to_quadrant(), then merge
    the partial results with Quadrant.update().
    """

    side: Laterality
    si: Optional[float] = None
    ml: Optional[float] = None
    depth: Optional[float] = None

    def to_quadrant(self) -> Quadrant:
        return Quadrant(
            side=self.side,
            si=_quantize_si(self.si) if self.si is not None else SiAxis.UNKNOWN,
            ml=_quantize_ml(self.ml) if self.ml is not None else MlAxis.UNKNOWN,
            depth=_quantize_depth(self.depth) if self.depth is not None else DepthAxis.UNKNOWN,
        )

"""
loc codes:
OU / L:     lateral     NA          NA
IN / D:     medial      NA          NA
Z:          medial      inferior    NA
X:          medial      superior    NA
Y:          lateral     inferior    NA
W:          lateral     superior    NA
C:          central     central     NA
A / T:      NA          superior    posterior
UP / U:     NA          superior    NA
LO / I:     NA          inferior    NA
S:          central     central     anterior
AN:         NA          NA          anterior
MD:         NA          NA          middle


depth codes:
A:          NA          NA          anterior
M:          NA          NA          middle
P:          NA          NA          posterior


clock codes:
-medial/lateral-
L: "1", "2", "3", "4", "5" = lateral
L: "7", "8", "9", "10", "11" = medial
R: "7", "8", "9", "10", "11" = lateral
R: "1", "2", "3", "4", "5" = medial
_: "6", "12" = central

-superior/inferior-
_: "10", "11", "12", "1", "2" = superior
_: "3", "9" = central
_: "4", "5", "6", "7", "8" = inferior


from finding > quadrant
1. finding has some code describing a quadrant-position
2. .from_magview_codes() method maps it to the described quadrant (reconciling any divergent
   meanings, e.g. if the loc code has an associated depth but also an explicit depth code that
   disagrees — the explicit depth code takes priority)
3. parsed quadrant has an associated position in the coordinate space

from roi > quadrant
1. roi is centered on some region in the image
2. breast landmark geometry is used to establish where the center point is on the ML/SI/D axes
3. center position is captured as a RoiPosition (continuous normalized axis values)
4. RoiPosition.to_quadrant() maps to the closest discrete Quadrant

RoiPosition is the continuous-space analogue to LocCode/DepCode: just as those codes map to a
Quadrant via .to_quadrant(), an RoiPosition maps continuous axis measurements to a Quadrant.
Multi-view fusion (CC gives ML+SI, MLO gives SI+depth) works the same way as code reconciliation:
convert each view's RoiPosition to a partial Quadrant and merge with Quadrant.update().
"""


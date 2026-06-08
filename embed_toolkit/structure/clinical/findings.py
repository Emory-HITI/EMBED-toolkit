from enum import Enum
from dataclasses import dataclass

from abc import ABC

from embed_toolkit.structure.clinical.quadrants import Quadrant

"""

"finding": [
    "massshape", -- string -- "find_mass_shape"
    "massmargin", -- string -- "find_mass_margin"
    "massdens", -- string -- "find_mass_density"
    "calcfind", -- string -- "find_calc_morphology"
    "calcdistri", -- string -- "find_calc_distribution"
    "calcnumber", -- string -- "find_calc_number"
    "otherfind", -- string -- "find_other"
    "implanfind", -- string -- "find_implant"
    "consistent", -- string -- "find_consistent"
    "side", -- string -- "find_side"
    "size", -- string -- "find_size"
    "location", -- string -- "find_loc"
    "depth", -- string -- "find_depth"
    "distance", -- int -- "find_distance"
    "numfind", -- int -- "find_number"
    "asses", -- string -- "find_birads"
    "recc", -- string -- "find_recommendation"
    "stable", -- int/bool (currently 0/-1) -- "find_stable"
    "new", -- int/bool (currently 0/-1) -- "find_new"
    "changed", -- string -- "find_changed"
],

"""

class BiRads(Enum):
    ZERO = 0  # incomplete
    ONE = 1  # negative
    TWO = 2  # benign
    THREE = 3  # probably benign
    FOUR = 4  # suspicious
    FIVE = 5  # highly suggestive of malignancy
    SIX = 6  # known biopsy-proven malignancy


@dataclass
class Finding(ABC):
    num: int  # numfind
    assessment: BiRads
    quadrant: Quadrant


# class Finding:
#     def __init__(
#         self,
#         id: int,
#         laterality: Laterality,
#         loc_codes: Optional[list[str]] = None,
#         depth_codes: Optional[list[str]] = None,
#     ):
#         self.hash_id: str = uuid.uuid4().hex
#         self.id: int = id
#         self.laterality: Laterality = laterality
#         self.loc_codes: list[str] = loc_codes if loc_codes is not None else []
#         self.depth_codes: list[str] = depth_codes if depth_codes is not None else []
#
#     def __repr__(self) -> str:
#         return f"Finding({self.id}, side: {self.laterality.value}, locs: {self.loc_codes}, depths: {self.depth_codes})"
#
#     def evaluate_quadrant(
#         self,
#         laterality: Optional[Laterality] = None,
#     ) -> Optional[Quadrant]:
#         # use an explicit override (e.g. when resolving a BILATERAL finding for
#         # a specific side's image) or fall back to the finding's own laterality
#         side = laterality if laterality is not None else self.laterality
#
#         result = Quadrant(side=side)
#         found_any = False
#
#         # depth codes are processed first so their explicit depth value takes
#         # priority over any depth implied by a loc code (update() never
#         # overwrites a known axis)
#         for code_str in self.depth_codes:
#             try:
#                 result.update(DepCode(code_str).to_quadrant(side))
#                 found_any = True
#             except ValueError:
#                 pass
#
#         for code_str in self.loc_codes:
#             try:
#                 loc_code = _coerce_loc_code(code_str)
#                 result.update(loc_code.to_quadrant(side))
#                 found_any = True
#             except ValueError:
#                 pass
#
#         return result if found_any else None
#
#     def __hash__(self) -> int:
#         return hash(self.hash_id)
#
# Calcifications -----------------------------------------------------------------------------------------------


class CalcMorphology(Enum):
    SKIN = "skin"
    VASCULAR = "vascular"
    COARSE = "coarse"
    LARGE_ROD_LIKE = "large rod-like"
    ROUND = "round"
    RIM = "rim"
    DYSTROPHIC = "dystrophic"
    MILK_OF_CALCIUM = "milk of calcium"
    SUTURE = "suture"
    AMORPHOUS = "amorphous"
    COARSE_HETERO = "coarse heterogenous"
    FINE_PLEOMORPH = "fine pleomorph"
    FINE_LINEAR = "fine linear"
    UNKNOWN = "unknown"


class CalcDistribution(Enum):
    DIFFUSE = "diffuse"
    REGIONAL = "regional"
    GROUPED = "grouped"
    LINEAR = "linear"
    SEGMENTAL = "segmental"
    UNKNOWN = "unknown"


@dataclass
class CalcFinding(Finding):
    morphology: CalcMorphology = CalcMorphology.UNKNOWN
    distribution: CalcDistribution = CalcDistribution.UNKNOWN

    def __repr__(self) -> str:
        return f"CalcFinding(BI-RADs: {self.assessment}, {self.morphology} + {self.distribution})"


# --------------------------------------------------------------------------------------------------------------
#
#
# Masses -------------------------------------------------------------------------------------------------------


class MassShape(Enum):
    OVAL = "oval"
    ROUND = "round"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


class MassMargin(Enum):
    CIRCUMSCRIBED = "circumscribed"
    OBSCURED = "obscured"
    MICROLOBULATED = "microlobulated"
    INDISTINCT = "indistinct"
    SPICULATED = "spiculated"
    UNKNOWN = "unknown"


class MassDensity(Enum):
    HIGH = "high"
    EQUAL = "equal"
    LOW = "low"
    FAT = "fat containing"
    UNKNOWN = "unknown"


@dataclass
class MassFinding(Finding):
    shape: MassShape = MassShape.UNKNOWN
    margin: MassMargin = MassMargin.UNKNOWN
    density: MassDensity = MassDensity.UNKNOWN

    def __repr__(self) -> str:
        return f"MassFinding(BI-RADs: {self.assessment}, {self.shape} + {self.margin} + {self.density})"


# --------------------------------------------------------------------------------------------------------------
#
#
# Architectural Distortions ------------------------------------------------------------------------------------


@dataclass
class ArchDistFinding(Finding):
    def __repr__(self) -> str:
        return f"ArchDistFinding(BI-RADs: {self.assessment})"


# --------------------------------------------------------------------------------------------------------------
#
#
# Asymmetries --------------------------------------------------------------------------------------------------


class AsymType(Enum):
    NOS = "NOS"
    GLOBAL = "global"
    FOCAL = "focal"
    DEVELOPING = "developing"


@dataclass
class AsymFinding(Finding):
    type: AsymType = AsymType.NOS

    def __repr__(self) -> str:
        return f"AsymFinding(BI-RADs: {self.assessment}, {self.type})"


# --------------------------------------------------------------------------------------------------------------

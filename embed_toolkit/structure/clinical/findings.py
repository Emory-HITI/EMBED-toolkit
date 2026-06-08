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

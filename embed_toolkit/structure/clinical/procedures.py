from datetime import date
from enum import Enum

from embed_toolkit.elements.general import Laterality


class ProcedureType(Enum):
    BIOPSY = "B"
    SURGERY = "S"

class PathSeverity(Enum):
    INVASIVE_BC = 0.0
    IN_SITU_BC = 1.0
    HIGH_RISK_LESION = 2.0
    BORDERLINE_LESION = 3.0
    BENIGN_LESION = 4.0
    NON_BREAST_CANCER = 5.0


class Procedure:
    def __init__(self): 
        self.severity: PathSeverity
        self.type: ProcedureType
        self.proc_date_anon: date
        self.laterality: Laterality
        self.path_codes: list[str] # path1-10



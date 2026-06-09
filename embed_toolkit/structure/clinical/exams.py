from enum import Enum
from datetime import date
from abc import ABC


class BreastDensity(Enum):
    # TODO: move to general concepts location
    A = 1.0
    B = 2.0
    C = 3.0
    D = 4.0
    MALE = 5.0

class Exam(ABC):
    def __init__(self):
        self.acc_anon: int
        self.date_anon: date
        self.desc: str # study description

        self.tissue_density: BreastDensity
        self.patient_age: float
        self.patient_first_3_zip: int

        self.visit_type: str # vtype
        self.site_id: str # from loc_num
        self.tech_init: str # do we actually need this? i don't think so?

        self.init: str # what actually is this...?
        self.proccode: str # and this..? 
        self.special_case_type: str # case

        # derived, maybe just re-count findings instead
        self.total_L_find: int
        self.total_R_find: int


class ScreeningExam(Exam):
    def __repr__(self) -> str:
        return f"ScreeningExam({self.date_anon} - {self.desc})"

class DiagnosticExam(Exam):
    def __repr__(self) -> str:
        return f"DiagnosticExam({self.date_anon} - {self.desc})"

class OtherExam(Exam):
    def __repr__(self) -> str:
        return f"OtherExam({self.date_anon} - {self.desc})"


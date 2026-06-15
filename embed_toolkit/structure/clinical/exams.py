from enum import Enum
from datetime import date
from dataclasses import dataclass
import pandas as pd
from abc import ABC, abstractmethod

from embed_toolkit.structure.clinical.patients import (
    Patient,
    PatientGender,
    PatientDemographics,
    PatientRace,
    PatientEthnicity,
)

# exam-level enums -------------------------------------------------------------------------------------


class ExamType(Enum):
    SCREENING = "screening"
    DIAGNOSTIC = "diagnostic"
    OTHER = "other"
    INVALID = "invalid"

    @classmethod
    def _missing_(cls, value) -> "ExamType":
        return cls.OTHER


class BreastDensity(Enum):
    # TODO: move to general concepts location
    A = 1.0
    B = 2.0
    C = 3.0
    D = 4.0
    MALE = 5.0
    UNKNOWN = float("nan")

    @classmethod
    def _missing_(cls, value) -> "BreastDensity":
        return cls.UNKNOWN


class VisitType(Enum):
    PROCEDURE = "1"
    LOC = "2"
    RESPONSE_TO_TREATMENT = "3"
    MR_FIND_SHORT_TERM_FOLLOW_UP = "6"
    ADDITIONAL_EVAL_FROM_RECENT_STUDY = "A"
    POST_BIOPSY = "B"
    TECHNICAL_CALLBACK = "C"
    OUTSIDE_FILMS_NOT_AVAILABLE = "D"
    EVAL_FINDING_ON_OSF = "E"
    FOLLOW_UP_SHORT_INTERVAL = "F"
    EXTENT_OF_DISEASE = "G"
    HIGH_RISK_SCREENING = "H"
    EVAL_FINDING_ON_MRI = "I"
    ABNORMAL_FINDING_ON_PRIOR_STUDY = "K"
    POST_LUMPECTOMY = "L"
    POST_MASTECTOMY_FOLLOW_UP = "M"
    SPECIMEN = "N"
    REVIEW_OUTSIDE_STUDY = "O"
    PROBLEM_INDICATED = "P"
    REFLECTOR_PLACEMENT = "Q"
    PRE_REDUCTION_MAMMOPLASTY = "R"
    SCREENING = "S"
    PRE_RADIATION_THERAPY = "T"
    IMPLANT_EVALUATION = "U"
    ADDITIONAL_EVAL_FROM_CURRENT_SCREENING = "V"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value) -> "VisitType":
        return cls.UNKNOWN


class SpecialCaseType(Enum):
    INTERESTING_CASE = "I"
    TEACHING_FILE = "T"
    TOMO_STUDY = "O"
    BREAST_CT_NON_CONTRAST = "N"
    BREAST_CT_CONSTRAST = "C"
    QA_ALERT = "Q"
    NONE = "none"

    @classmethod
    def _missing_(cls, value) -> "SpecialCaseType":
        return cls.NONE


# ------------------------------------------------------------------------------------------------------

# exam-level demographics dataclass --------------------------------------------------------------------


@dataclass(frozen=True)
class PatientExamDemographics:
    # patient demographics which can vary over time
    age: float
    first_3_zip: int

    gender: PatientGender
    # marital_status: PatientMaritalStatus

    # attr for static demographics
    _static: PatientDemographics

    @property
    def dob(self) -> date:
        # wrapper property to expose the dob attr from the underlying static demographics
        return self._static.dob

    @property
    def race(self) -> PatientRace:
        # wrapper property to expose the race attr from the underlying static demographics
        return self._static.race

    @property
    def ethnicity(self) -> PatientEthnicity:
        # wrapper property to expose the ethnicity attr from the underlying static demographics
        return self._static.ethnicity

    @classmethod
    def from_series(
        cls, data: pd.Series, patient: "Patient"
    ) -> "PatientExamDemographics":
        return cls(
            age=float(data.age_at_study),
            first_3_zip=int(data.first_3_zip),
            gender=PatientGender(data.GENDER_DESC),
            _static=patient.demographics,
        )


# ------------------------------------------------------------------------------------------------------

# exam core object -------------------------------------------------------------------------------------


class Exam(ABC):
    def __init__(self, data: pd.Series, patient: Patient):
        self.patient: Patient = patient
        self.demographics: PatientExamDemographics = (
            PatientExamDemographics.from_series(data, patient)
        )
        self.acc_anon: int = int(data["acc_anon"])
        self.date_anon: date = (
            data["studydate_anon"].to_pydatetime().date()
        )  # must be input as a pandas datetime col
        self.desc: str = str(data["desc"])
        self.tissue_density: BreastDensity = BreastDensity(data["tissueden"])
        self.patient_age: float = float(data["age_at_study"])
        self.patient_first_3_zip: int = int(data["first_3_zip"])
        self.visit_type: VisitType = VisitType(data["vtype"])
        self.special_case_type: SpecialCaseType = SpecialCaseType(data["case"])
        self.site_id: str = str(data["loc_num_anon"])

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @property
    @abstractmethod
    def exam_type(self) -> ExamType:
        return ExamType.INVALID

    def __hash__(self) -> int:
        return hash(self.acc_anon)

    @classmethod
    def _resolve_subclass(cls, data: pd.Series) -> type["Exam"]:
        """Determine the correct Exam subclass from row values."""
        match ExamType(data.mg_exam_type):
            case ExamType.SCREENING:
                return ScreeningExam
            case ExamType.DIAGNOSTIC:
                return DiagnosticExam
            case _:
                return OtherExam

    @classmethod
    def from_series(cls, data: pd.Series, patient: Patient) -> "Exam":
        """
        Factory: parse a DataFrame row into the appropriate Exam subtype.
        Can be called on the base class — Exam.from_series(row) — or on any
        subclass directly if you want to bypass dispatch and force a type.
        """
        target_cls = cls if cls is not Exam else cls._resolve_subclass(data)

        return target_cls(data, patient)


class ScreeningExam(Exam):
    # def __init__(self, data: pd.Series, patient: Patient) -> None:
    #     # only needed if we have custom subclass logic outside the abc
    #     super().__init__(data, patient)

    @property
    def exam_type(self) -> ExamType:
        return ExamType.SCREENING

    def __repr__(self) -> str:
        return f"ScreeningExam([{self.acc_anon}] {self.date_anon} - {self.desc})"


class DiagnosticExam(Exam):
    # def __init__(self, data: pd.Series, patient: Patient) -> None:
    #     # only needed if we have custom subclass logic outside the abc
    #     super().__init__(data, patient)

    @property
    def exam_type(self) -> ExamType:
        return ExamType.DIAGNOSTIC

    def __repr__(self) -> str:
        return f"DiagnosticExam([{self.acc_anon}] {self.date_anon} - {self.desc})"


class OtherExam(Exam):
    # def __init__(self, data: pd.Series, patient: Patient) -> None:
    #     # only needed if we have custom subclass logic outside the abc
    #     super().__init__(data, patient)

    @property
    def exam_type(self) -> ExamType:
        return ExamType.OTHER

    def __repr__(self) -> str:
        return f"OtherExam([{self.acc_anon}] {self.date_anon} - {self.desc})"

from datetime import date
from enum import Enum
from dataclasses import dataclass


class PatientRace(Enum):
    ASIAN = "asian"
    BLACK = "black"
    WHITE = "white"
    OTHER = "other"
    UNKNOWN = "unknown"


class PatientEthnicity(Enum):
    HISPANIC = "hispanic or latino"
    NOT_HISPANIC = "not hispanic or latino"
    UNKNOWN = "unknown"

class PatientGender(Enum):
    FEMALE = "F"
    MALE = "M"
    OTHER = "X"
    UNKNOWN = "U"

class PatientMaritalStatus(Enum):
    # fill out with other levels after verifying
    SINGLE = "single"
    MARRIED = "married"

@dataclass
class PatientDemographics:
    dob: date
    race: PatientRace
    ethnicity: PatientEthnicity
    gender: PatientGender
    marital_status: PatientMaritalStatus


class Patient:
    def __init__(self, empi_anon: int, cohort_num: int, demographics: PatientDemographics):
        self.empi_anon: int = empi_anon
        self.cohort_num: int = cohort_num
        self.demographics: PatientDemographics = demographics

        # TODO: finish this c:

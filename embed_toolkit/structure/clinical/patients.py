from datetime import date
from enum import Enum
from dataclasses import dataclass
import pandas as pd


# patient demographics enums ---------------------------------------------------------------------------


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
    OTHER = "other"


class PatientLanguage(Enum):
    UNKNOWN_OR_MISSING = "unknown"
    ENGLISH = "english"
    SPANISH = "spanish"
    EAST_OR_SOUTHEAST_ASIAN = "e_or_se_asian"
    SOUTH_ASIAN = "s_asian"
    IRANIAN = "iranian"
    EAST_AFRICAN_OR_HORN_OF_AFRICA = "e_africa_or_horn_of_africa"
    WEST_OR_CENTRAL_AFRICAN = "w_or_c_african"
    EUROPEAN = "european"
    SEMITIC = "semitic"
    KAREN_OR_TIBETO_BURMAN = "karen_or_tibeto_burman"  # refugee community, clinically meaningful to keep distinct
    PACIFIC_OR_OCEANIC = "pacific_or_oceanic"
    INDIGENOUS_AMERICAS = "indigenous_americas"
    SIGN_LANGUAGE = "sign_language"
    HAITIAN = "haitian"
    OTHER = (
        "other"  # catches values with no parsed entry -- should ideally never see this!
    )

    @classmethod
    def _missing_(cls, value) -> "PatientLanguage":
        match str(value).lower():
            case "english":
                return cls.ENGLISH
            case "spanish":
                return cls.SPANISH
            case "haitian":
                return cls.HAITIAN
            case "sign language":
                return cls.SIGN_LANGUAGE
            case value if value in [
                "french",
                "portuguese",
                "russian",
                "romanian",
                "dutch",
                "german",
                "croatian",
                "bosnian",
                "serbian",
                "ukrainian",
                "bulgarian",
                "italian",
                "polish",
                "greek",
                "albanian",
                "swedish",
                "danish",
                "norwegian nynorsk",
                "czech",
                "lithuanian",
                "irish",
                "faroese",
                "yiddish",
                "afrikaans",
                "corsican",
                "romani",
                "esperanto",
            ]:
                return cls.EUROPEAN
            case value if value in [
                "korean",
                "vietnamese",
                "chinese",
                "mandarin",
                "cantonese",
                "taiwanese hokkien",
                "hakka",
                "japanese",
                "thai",
                "cambodian",
                "central khmer",
                "lao",
                "burmese",
                "tagalog",
                "filipino",
                "indonesian",
                "bahasa",
                "sundanese",
            ]:
                return cls.EAST_OR_SOUTHEAST_ASIAN
            case value if value in [
                "hindi",
                "gujarati",
                "bengali",
                "nepali",
                "urdu",
                "panjabi",
                "marathi",
                "sinhala",
                "tamil",
                "telugu",
                "malayalam",
                "indo-aryan",
            ]:
                return cls.SOUTH_ASIAN
            case value if value in [
                "yoruba",
                "akan",
                "twi",
                "ewe",
                "igbo",
                "bambara",
                "wolof",
                "hausa",
                "kanuri",
                "fulani",
                "mandinka",
                "edo",
                "efik",
                "krio",
            ]:
                return cls.WEST_OR_CENTRAL_AFRICAN
            case value if value in [
                "amharic",
                "somali",
                "tigrinya",
                "swahili",
                "oromo",
                "afar",
                "kinyarwanda",
                "rundi",
                "dinka",
                "zulu",
                "ovambo",
            ]:
                return cls.EAST_AFRICAN_OR_HORN_OF_AFRICA
            case value if value in ["persian", "dari", "pushto", "kurdish", "aramaic"]:
                return cls.IRANIAN
            case value if value in ["unknown", "other", "decline to answer"]:
                return cls.UNKNOWN_OR_MISSING
            case value if value in ["karen", "kareni", "zomi", "falam"]:
                return cls.KAREN_OR_TIBETO_BURMAN
            case value if value in ["quechua", "nahuatl", "guarani"]:
                return cls.INDIGENOUS_AMERICAS
            case value if value in ["fijian", "chuukese"]:
                return cls.PACIFIC_OR_OCEANIC
            case value if value in ["arabic", "hebrew"]:
                return cls.SEMITIC
            case _:
                return cls.OTHER


# ------------------------------------------------------------------------------------------------------

# patient demographics dataclasses ---------------------------------------------------------------------


@dataclass(frozen=True)
class PatientDemographics:
    # patient demographics that are reasonably fixed
    dob: date
    race: PatientRace
    ethnicity: PatientEthnicity
    language: PatientLanguage


@dataclass(frozen=True)
class PatientExamDemographics:
    # patient demographics which can vary over time
    age: float
    first_3_zip: int

    gender: PatientGender
    marital_status: PatientMaritalStatus

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


# ------------------------------------------------------------------------------------------------------

# patience core object ---------------------------------------------------------------------------------


class Patient:
    def __init__(
        self, empi_anon: int, cohort_num: int, demographics: PatientDemographics
    ):
        """
        what do we need to handle here?
        1. static patient demographics -- (DOB, race, ethnicity)
        2. empi_anon/cohort_num

        is there anything else totally outside lower data tiers?
        - risk scores are tied to time points -- best linked to exams
        - family cancer history etc.
        - brca and other genetic risk factors are at patient level (even if determined at a later time point)
        """
        self.empi_anon: int = empi_anon
        self.cohort_num: int = cohort_num
        self.demographics: PatientDemographics = demographics

        # TODO: finish this c:


def parse_patient_data(data: pd.Series):
    # parse patient demographics
    race: PatientRace = PatientRace("")


# ------------------------------------------------------------------------------------------------------

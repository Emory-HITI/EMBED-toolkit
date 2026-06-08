import uuid
from abc import ABC
from typing import Union

import pandas as pd
import pydicom

from embed_toolkit.elements.alignment import Alignment
from embed_toolkit.elements.general import Laterality
from embed_toolkit.structure.imaging.general import ImageModality, ViewPosition
from embed_toolkit.structure.imaging.rois import RegionOfInterest

# ─────────────────────────────────────────────────────────────────────────────
# ImageBase
# ─────────────────────────────────────────────────────────────────────────────


class Mammogram(ABC):
    """Base class for all image representations in the pipeline."""

    """
    TODO:
    what else should this base class be capable of handling?
    - basic pixel array preprocessing ops? or should this be a mixin???????
    - roi parsing/display stuff is good, but should that also be a mixin?

    potential mixins: RegionOfInterestMixin, PixelArrayMixin
    RegionOfInterestMixin:
    - parse roi str/other coords into RegionOfInterest objects
    - interacts with plotting Mixin to handle display?
    - adds methods for ROI transfer/resizing?

    PixelArrayMixin: -- though the entire plotting mixin basically requires this? maybe a PixelProcessingMixin instead to extend the base capabilities?
    - modular/composable pixel array preprocessing
    - image masking
    - breast boundary/contour extraction
    - pixel-based orientation detection??????

    
    """

    def __init__(
        self,
        file_path: str,
        laterality: Laterality,
        view_position: ViewPosition,
        height: float,
        width: float,
        alignment: Alignment,
        modality: ImageModality = ImageModality.UNKNOWN,
        frames: int = 1,
    ) -> None:
        self.hash_id: str = uuid.uuid4().hex
        self.path: str = file_path
        self.laterality: Laterality = laterality
        self.view_position: ViewPosition = view_position
        self.height: float = height
        self.width: float = width
        self.alignment: Alignment = alignment
        self.modality: ImageModality = modality
        self.frames: int = frames
        self.rois: list[RegionOfInterest] = []

    def __hash__(self) -> int:
        return hash(self.hash_id)

    # def transfer_rois(self, target: "ImageBase") -> list[RegionOfInterest]:
    #     """Transfer this image's ROIs to the target, resizing and re-aligning as needed."""
    #     return [roi.transfer(target=target) for roi in self.rois]

    @classmethod
    def from_dicom(
        cls,
        dicom: pydicom.FileDataset,
        modality: Union[str, ImageModality],
        file_path: str = "",
    ) -> "Mammogram":
        """Constructor that builds an ImageBase object from a loaded pydicom.FileDataset"""
        if not isinstance(modality, ImageModality):
            modality: ImageModality = ImageModality(modality)

        laterality: Laterality = Laterality(dicom[(0x20, 0x60)].repval)
        view_position: ViewPosition = ViewPosition(dicom[(0x18, 0x5101)].repval)
        alignment: Alignment = Alignment.from_orientation(
            dicom[(0x20, 0x20)].repval, laterality, view_position
        )

        height: int = int(dicom[0x28, 0x10].value)  # 'Rows' tag
        width: int = int(dicom[0x28, 0x11].value)  # 'Columns' tag
        frames: int = int(dicom[0x20, 0x1002].value)  # 'ImagesInAcquisition' tag

        return cls(
            file_path=file_path,
            laterality=laterality,
            view_position=view_position,
            height=height,
            width=width,
            alignment=alignment,
            modality=modality,
            frames=frames,
        )

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        laterality_col: str = "ImageLateralityFinal",
        view_pos_col: str = "ViewPosition",
        orientation_col: str = "PatientOrientation",
        modality_col: str = "FinalImageType",
        height_col: str = "Rows",
        width_col: str = "Columns",
        frames_col: str = "ImagesInAcquisition",
        path_col: str = "anon_dicom_path",
    ) -> "Mammogram":
        """Constructor that builds an ImageBase object from a Pandas series"""
        laterality: Laterality = Laterality(str(series[laterality_col]))
        view_position: ViewPosition = ViewPosition(str(series[view_pos_col]))
        alignment: Alignment = Alignment.from_orientation(
            str(series[orientation_col]), laterality, view_position
        )

        modality: ImageModality = ImageModality(str(series[modality_col]))

        height: int = int(series[height_col])
        width: int = int(series[width_col])
        frames: int = int(series[frames_col])

        return cls(
            file_path=str(series[path_col]),
            laterality=laterality,
            view_position=view_position,
            height=height,
            width=width,
            alignment=alignment,
            modality=modality,
            frames=frames,
        )
 


from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
import pydicom

from embed_toolkit.elements.general import (
    FovHFlip,
    FovRotation,
    Laterality,
    OrientationDirection,
    PatientOrientation,
    ViewPosition,
)


class AlignmentDirection(Enum):
    RIGHT = "Right"
    LEFT = "Left"
    UP = "Up"
    DOWN = "Down"
    UNKNOWN = "Unknown"

    @property
    def opposite(self):
        opposites = {
            AlignmentDirection.RIGHT: AlignmentDirection.LEFT,
            AlignmentDirection.LEFT: AlignmentDirection.RIGHT,
            AlignmentDirection.UP: AlignmentDirection.DOWN,
            AlignmentDirection.DOWN: AlignmentDirection.UP,
        }
        return opposites.get(self, AlignmentDirection.UNKNOWN)


@dataclass
class Alignment:
    x: AlignmentDirection  # alignment looking across rows from left to right
    y: AlignmentDirection  # alignment looking down cols from top to bottom

    @property
    def _reference(self) -> "Alignment":
        return Alignment(AlignmentDirection.LEFT, AlignmentDirection.UP)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Alignment(X: {self.x.value}, Y: {self.y.value})"

    @staticmethod
    def _flip_required(source: AlignmentDirection, target: AlignmentDirection) -> bool:
        if target == AlignmentDirection.UNKNOWN:
            raise ValueError("Target alignment should not have an 'UNKNOWN' direction!")
        # we only need to flip if the source direction is the opposite of the target
        return source == target.opposite

    def _prepare_target(self, target: Optional["Alignment"]) -> "Alignment":
        target = target or self._reference

        # Check if axes are crossed (e.g. Target X is Up/Down), implying rotation
        if target.x in {AlignmentDirection.UP, AlignmentDirection.DOWN} or target.y in {
            AlignmentDirection.LEFT,
            AlignmentDirection.RIGHT,
        }:
            raise NotImplementedError("Rotation required! Implement method.")

        return target

    def realign_coords(
        self,
        coords: tuple[float, float, float, float],
        height: float,
        width: float,
        target_alignment: Optional["Alignment"] = None,
    ) -> tuple[float, float, float, float]:
        # provide the reference as default if no target_alignment was given
        target_alignment: Alignment = self._prepare_target(target_alignment)

        # unpack coords
        y_min, x_min, y_max, x_max = coords

        # check if left/right flipping is needed
        if self._flip_required(self.x, target_alignment.x):
            x_min, x_max = width - x_max, width - x_min

        # check if up/down flipping is needed
        if self._flip_required(self.y, target_alignment.y):
            y_min, y_max = height - y_max, height - y_min

        return (y_min, x_min, y_max, x_max)

    def realign_image(
        self,
        image: np.ndarray,
        target_alignment: Optional["Alignment"] = None,
    ) -> np.ndarray:
        # provide the reference as default if no target_alignment was given
        target_alignment: Alignment = self._prepare_target(target_alignment)

        # check if left/right flipping is needed
        if self._flip_required(self.x, target_alignment.x):
            image = image[..., ::-1]  # flip last axis

        # check if up/down flipping is needed
        if self._flip_required(self.y, target_alignment.y):
            image = image[..., ::-1, :]  # flip second-to-last axis

        return image

    @classmethod
    def from_dicom(cls, dicom: pydicom.FileDataset) -> "Alignment":
        orientation: PatientOrientation = PatientOrientation(dicom[(0x20, 0x20)].repval)
        laterality: Laterality = Laterality(dicom[(0x20, 0x60)].repval)
        view_position: ViewPosition = ViewPosition(dicom[(0x18, 0x5101)].repval)

        return cls.from_orientation(orientation, laterality, view_position)

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        orientation_col: str = "PatientOrientation",
        laterality_col: str = "ImageLateralityFinal",
        view_pos_col: str = "ViewPosition",
    ) -> "Alignment":
        orientation: PatientOrientation = PatientOrientation(
            str(series[orientation_col])
        )
        laterality: Laterality = Laterality(str(series[laterality_col]))
        view_position: ViewPosition = ViewPosition(str(series[view_pos_col]))

        return cls.from_orientation(orientation, laterality, view_position)

    @classmethod
    def from_orientation(
        cls,
        orientation: Union[str, tuple[str, str], PatientOrientation],
        laterality: Union[str, Laterality],
        view_position: Union[str, ViewPosition],
    ) -> "Alignment":
        # parse any inputs that have not been provided as Enums
        if not isinstance(orientation, PatientOrientation):
            orientation: PatientOrientation = PatientOrientation(orientation)
        if not isinstance(laterality, Laterality):
            laterality: Laterality = Laterality(laterality)
        if not isinstance(view_position, ViewPosition):
            view_position: ViewPosition = ViewPosition(view_position)

        def _check_alignment(
            laterality: Laterality,
            view_position: ViewPosition,
            direction: OrientationDirection,
        ) -> AlignmentDirection:
            # internal function to convert an OrientationDirection to an AlignmentDirection given the image laterality/view
            match (laterality, view_position, direction):
                # all views -----------------------------------------------\
                case (_, _, OrientationDirection.P):
                    return AlignmentDirection.RIGHT
                case (_, _, OrientationDirection.A):
                    return AlignmentDirection.LEFT
                # LCC views -----------------------------------------------\
                case (Laterality.LEFT, ViewPosition.CC, OrientationDirection.L):
                    return AlignmentDirection.DOWN
                case (Laterality.LEFT, ViewPosition.CC, OrientationDirection.R):
                    return AlignmentDirection.UP
                # RCC views -----------------------------------------------\
                case (Laterality.RIGHT, ViewPosition.CC, OrientationDirection.L):
                    return AlignmentDirection.UP
                case (Laterality.RIGHT, ViewPosition.CC, OrientationDirection.R):
                    return AlignmentDirection.DOWN
                # LMLO views ----------------------------------------------\
                case (Laterality.LEFT, ViewPosition.MLO, OrientationDirection.FR):
                    return AlignmentDirection.UP
                case (Laterality.LEFT, ViewPosition.MLO, OrientationDirection.HL):
                    return AlignmentDirection.DOWN
                # RMLO views ----------------------------------------------\
                case (Laterality.RIGHT, ViewPosition.MLO, OrientationDirection.FL):
                    return AlignmentDirection.UP
                case (Laterality.RIGHT, ViewPosition.MLO, OrientationDirection.HR):
                    return AlignmentDirection.DOWN
                # ML views ------------------------------------------------\
                case (_, ViewPosition.ML, OrientationDirection.F):
                    return AlignmentDirection.UP
                case (_, ViewPosition.ML, OrientationDirection.H):
                    return AlignmentDirection.DOWN
                # LM views ------------------------------------------------\
                case (_, ViewPosition.LM, OrientationDirection.F):
                    return AlignmentDirection.UP
                case (_, ViewPosition.LM, OrientationDirection.H):
                    return AlignmentDirection.DOWN
                # LXCCL views ------------------------------------------------\
                case (Laterality.LEFT, ViewPosition.XCCL, OrientationDirection.R):
                    return AlignmentDirection.UP
                case (Laterality.LEFT, ViewPosition.XCCL, OrientationDirection.L):
                    return AlignmentDirection.DOWN
                # RXCCL views ------------------------------------------------\
                case (Laterality.RIGHT, ViewPosition.XCCL, OrientationDirection.L):
                    return AlignmentDirection.UP
                case (Laterality.RIGHT, ViewPosition.XCCL, OrientationDirection.R):
                    return AlignmentDirection.DOWN
                # unknown combinations ------------------------------------\
                case _:
                    return AlignmentDirection.UNKNOWN

        return cls(
            _check_alignment(laterality, view_position, orientation.row),
            _check_alignment(laterality, view_position, orientation.col),
        )

    @classmethod
    def from_fov(
        cls,
        rotation: Union[str, float, FovRotation],
        h_flip: Union[str, FovHFlip],
        laterality: Union[str, Laterality],
    ) -> "Alignment":
        # TODO: currently this method is incorrect for LM images -- this is the only image
        # type I've tried where this breaks
        # I either need to find another element that directly captures this difference, or I need
        # to require ViewPosition as an input to this

        # parse any inputs that have not been provided as Enums
        if not isinstance(rotation, FovRotation):
            rotation: FovRotation = FovRotation(float(rotation))
        if not isinstance(h_flip, FovHFlip):
            h_flip: FovHFlip = FovHFlip(h_flip)
        if not isinstance(laterality, Laterality):
            laterality: Laterality = Laterality(laterality)

        match (laterality, rotation, h_flip):
            # laterality: L
            case (Laterality.LEFT, FovRotation.FULL, FovHFlip.YES):
                return Alignment(AlignmentDirection.RIGHT, AlignmentDirection.UP)
            case (Laterality.LEFT, FovRotation.FULL, FovHFlip.NO):
                return Alignment(AlignmentDirection.LEFT, AlignmentDirection.UP)
            case (Laterality.LEFT, FovRotation.NONE, FovHFlip.YES):
                return Alignment(AlignmentDirection.LEFT, AlignmentDirection.DOWN)
            case (Laterality.LEFT, FovRotation.NONE, FovHFlip.NO):
                return Alignment(AlignmentDirection.RIGHT, AlignmentDirection.DOWN)
            # laterality: R
            case (Laterality.RIGHT, FovRotation.FULL, FovHFlip.YES):
                return Alignment(AlignmentDirection.RIGHT, AlignmentDirection.DOWN)
            case (Laterality.RIGHT, FovRotation.FULL, FovHFlip.NO):
                return Alignment(AlignmentDirection.LEFT, AlignmentDirection.DOWN)
            case (Laterality.RIGHT, FovRotation.NONE, FovHFlip.YES):
                return Alignment(AlignmentDirection.LEFT, AlignmentDirection.UP)
            case (Laterality.RIGHT, FovRotation.NONE, FovHFlip.NO):
                return Alignment(AlignmentDirection.RIGHT, AlignmentDirection.UP)
            case _:
                return Alignment(AlignmentDirection.UNKNOWN, AlignmentDirection.UNKNOWN)

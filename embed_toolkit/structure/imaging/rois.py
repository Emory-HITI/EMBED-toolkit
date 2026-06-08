from dataclasses import dataclass
from functools import cached_property
from typing import Optional


from embed_toolkit.elements.alignment import Alignment
from embed_toolkit.structure.imaging.base import ImageBase


@dataclass(frozen=True, eq=True)
class RegionOfInterest:
    coords: tuple[float, float, float, float]
    source: ImageBase
    frames: Optional[tuple[int, ...]] = None
    origin_roi: Optional["RegionOfInterest"] = None
    depth_derived: bool = False

    # _z_padding: ClassVar[int] = 3  # n adjacent frames to pad the depth region across

    @cached_property
    def area(self) -> float:
        # returns the area of the ROI
        return (self.y_max - self.y_min) * (self.x_max - self.x_min)

    @cached_property
    def centroid(self) -> tuple[float, float]:
        # returns the centroid of the ROI as an (x, y) tuple
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    # @cached_property
    # def z_min(self) -> Optional[int]:
    #     if self.frames is None:
    #         return None
    #     # clip the min z index - padding to 0
    #     return max(min(self.frames) - self._z_padding, 0)
    #
    # @cached_property
    # def z_max(self) -> Optional[int]:
    #     if self.frames is None:
    #         return None
    #     # it doesn't matter if this exceeds the max n frames in the
    #     # source, so we won't bother clipping it
    #     return max(self.frames) + self._z_padding

    def __repr__(self) -> str:
        frame_str: str = (
            f", frames: {self.z_min} - {self.z_max}" if self.z_min is not None else ""
        )
        return f"RegionOfInterest({self.coords}{frame_str})"

    @property
    def y_min(self) -> float:
        return self.coords[0]

    @property
    def x_min(self) -> float:
        return self.coords[1]

    @property
    def y_max(self) -> float:
        return self.coords[2]

    @property
    def x_max(self) -> float:
        return self.coords[3]

    # def _z_overlap(self, target: "RegionOfInterest") -> bool:
    #     """Returns False only when both ROIs have defined Z ranges that do not overlap."""
    #     if None in (self.z_min, self.z_max, target.z_min, target.z_max):
    #         return True
    #     return max(self.z_min, target.z_min) <= min(self.z_max, target.z_max)  # type: ignore[arg-type]

    def resize(
        self,
        target_shape: tuple[float, float],
        coords: Optional[tuple[float, float, float, float]] = None,
    ) -> tuple[float, float, float, float]:
        # use self.coords if none were provided
        if coords is None:
            coords: tuple[float, float, float, float] = self.coords

        # destructure the coords
        y_min, x_min, y_max, x_max = coords

        # get source and target image heights/widths
        source_height, source_width = self.source.height, self.source.width
        target_height, target_width = target_shape

        # divide each coord by the source dim and multiply by the target dim
        return (
            (y_min / source_height) * target_height,
            (x_min / source_width) * target_width,
            (y_max / source_height) * target_height,
            (x_max / source_width) * target_width,
        )

    def realign(
        self,
        target_alignment: Alignment,
        coords: Optional[tuple[float, float, float, float]] = None,
    ) -> tuple[float, float, float, float]:
        # use self.coords if none were provided
        if coords is None:
            coords: tuple[float, float, float, float] = self.coords

        return self.source.alignment.realign_coords(
            coords, self.source.height, self.source.width, target_alignment
        )

    # def transfer(self, target: ImageBase) -> "RegionOfInterest":
    #     # transfers an ROI to match the alignment and coordinate space
    #     # of a target image, returns a new ROI linked to the target image
    #     coords: tuple[float, float, float, float] = self.coords
    #
    #     # re-align the coords to match the target
    #     target_alignment: Alignment = target.alignment
    #     coords = self.realign(target_alignment=target_alignment, coords=coords)
    #
    #     # resize the coords to match the target
    #     target_shape: tuple[float, float] = (target.height, target.width)
    #     coords = self.resize(target_shape=target_shape, coords=coords)
    #
    #     return RegionOfInterest(
    #         coords,
    #         target,
    #         self.frames,
    #         origin_roi=self,
    #         depth_derived=self.depth_derived,
    #     )

    # def calc_iou(self, target: "RegionOfInterest") -> float:
    #     """
    #     Calculates 2D IoU between self and the target ROI if Z-ranges overlap. If
    #     either ROI does not have defined frames, Z-overlap is not considered.
    #     Returns: float (0.0 to 1.0)
    #     """
    #     if not self._z_overlap(target):
    #         return 0.0
    #
    #     inter_y1 = max(self.y_min, target.y_min)
    #     inter_x1 = max(self.x_min, target.x_min)
    #     inter_y2 = min(self.y_max, target.y_max)
    #     inter_x2 = min(self.x_max, target.x_max)
    #
    #     inter_h = max(0, inter_y2 - inter_y1)
    #     inter_w = max(0, inter_x2 - inter_x1)
    #
    #     if inter_h == 0 or inter_w == 0:
    #         return 0.0
    #
    #     inter_area = inter_h * inter_w
    #     union_area = self.area + target.area - inter_area
    #     return inter_area / (union_area + 1e-6)
    #
    # def calc_containment_ratio(self, target: "RegionOfInterest") -> float:
    #     """
    #     Calculates the fraction of 'self' that is spatially contained by 'target'.
    #     This is an asymmetric operation.
    #
    #     Formula: Intersection_Area / Area_of_Self
    #
    #     Returns: float (0.0 to 1.0)
    #         - 1.0 means 'self' is completely inside 'target'.
    #         - 0.0 means 'self' does not overlap 'target' at all.
    #     """
    #     if not self._z_overlap(target):
    #         return 0.0
    #
    #     inter_y1 = max(self.y_min, target.y_min)
    #     inter_x1 = max(self.x_min, target.x_min)
    #     inter_y2 = min(self.y_max, target.y_max)
    #     inter_x2 = min(self.x_max, target.x_max)
    #
    #     inter_h = max(0, inter_y2 - inter_y1)
    #     inter_w = max(0, inter_x2 - inter_x1)
    #
    #     if inter_h == 0 or inter_w == 0:
    #         return 0.0
    #
    #     inter_area = inter_h * inter_w
    #     # divide by self.area: how much of *this* ROI is covered (asymmetric)
    #     return inter_area / (self.area + 1e-6)

    # def calc_l2_norm(self, target: "RegionOfInterest") -> float:
    #     """
    #     Calculate the center-to-center distance between self and the target ROI.
    #     Returns np.inf if the ROIs do not overlap along the Z axis
    #     """
    #     if not self._z_overlap(target):
    #         return np.inf
    #
    #     s_x, s_y = self.centroid
    #     t_x, t_y = target.centroid
    #     return math.sqrt((t_x - s_x) ** 2 + (t_y - s_y) ** 2)

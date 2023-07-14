from __future__ import annotations

import enum
import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

import dungeon_maps as dmap
from dungeon_maps import MapBuilder, MapProjector
from dungeon_maps.maps import TopdownMap
from dungeon_maps.utils import Reduction


@enum.unique
class Occupancy(int, enum.Enum):
    unexplored = -1
    free = 0
    occupied = 1


class TopDownMapper:
    def __init__(self, builder: MapBuilder):
        self.builder = builder

    @classmethod
    def create_occ_builder(
        cls,
        map_res: float,
        local_map_width: int,
        local_map_height: int,
        unexplored: int = Occupancy.unexplored,
        reduction: Reduction = Reduction.replace,
    ):
        """
        This is a lazy constructor if you plan to only want to merge occupancy maps into a global map We initialize camera parameters to dummy values since they are not used for merging.
        """
        global_proj = dmap.MapProjector(
            fill_value=unexplored,
            to_global=True,
            # Any non-zero value will do for initial width and height
            cam_pose=None,
            map_width=local_map_width,
            map_height=local_map_height,
            # Dummy values for camera parameters
            width=100,
            height=100,
            hfov=np.deg2rad(90),
            vfov=None,
            width_offset=0.0,
            height_offset=0.0,
            cam_pitch=np.deg2rad(0),
            cam_height=1.0,
            map_res=map_res,  # meters per pixel
            reduction=reduction,
        )
        builder = MapBuilder(global_proj)
        return cls(builder)

    @classmethod
    def create_semantic_mapper(
        cls,
        camera_hw: Tuple[int, int],
        hfov: float,
        trunc_depth_min: float,
        trunc_depth_max: float,
        map_res: float,
        cam_pose: Optional[Tensor],
        cam_pitch: Optional[Tensor],
        cam_height: Optional[Tensor],
        reduction: Reduction = Reduction.replace,
        fill_value: float | int = 0,
        trunc_height_max: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        map_width = int(2 * int(trunc_depth_max // map_res) * np.tan(0.5 * hfov))
        map_height = int(trunc_depth_max // map_res)
        proj = MapProjector(
            width=camera_hw[1],
            height=camera_hw[0],
            hfov=hfov,
            vfov=None,
            cam_pose=cam_pose,
            width_offset=map_width // 2,
            height_offset=0,
            cam_pitch=cam_pitch,
            cam_height=cam_height,
            map_res=map_res,
            map_width=map_width,
            map_height=map_height,
            trunc_depth_min=trunc_depth_min,
            trunc_depth_max=trunc_depth_max,
            reduction=reduction,
            fill_value=fill_value,
            to_global=True,
            device=device,
            trunc_height_max=trunc_height_max,
        )
        builder = MapBuilder(proj)
        return cls(builder)

    def reset(
        self,
        cam_pose: Tensor,
        idx: Optional[Sequence[int]] = None,
        cam_pitch: Optional[Tensor] = None,
        cam_height: Optional[Tensor] = None,
    ) -> Tensor:
        if idx is None:
            idx = torch.arange(cam_pose.shape[0])
        prev_pose = self.builder.proj.cam_pose
        if prev_pose is None or len(idx) == prev_pose.shape[0]:
            self.builder._proj = self.builder.proj.clone(
                cam_pose=cam_pose, cam_pitch=cam_pitch, cam_height=cam_height
            )
            self.builder._world_map = TopdownMap(
                map_projector=self.builder.proj.clone()
            )
        else:
            origin = self.builder.proj.cam_pose.clone()  # (b, 3)
            origin[idx] = cam_pose
            initial_height = self.builder.proj.cam_height
            if cam_height is not None:
                initial_height = self.builder.proj.cam_height.clone()
                initial_height[idx] = cam_height
            initial_pitch = self.builder.proj.cam_pitch
            if cam_pitch is not None:
                initial_pitch = self.builder.proj.cam_pitch.clone()
                initial_pitch[idx] = cam_pitch

            self.builder._proj = self.builder.proj.clone(
                cam_pose=origin, cam_pitch=initial_pitch, cam_height=initial_height
            )

            current_pose = self.builder._world_map.proj.cam_pose.clone()
            current_pose[idx] = cam_pose

            current_height = self.builder._world_map.proj.cam_height
            if cam_height is not None:
                current_height = self.builder._world_map.proj.cam_height.clone()
                current_height[idx] = cam_height
            current_pitch = self.builder._world_map.proj.cam_pitch
            if cam_pitch is not None:
                current_pitch = self.builder._world_map.proj.cam_pitch.clone()
                current_pitch[idx] = cam_pitch

            self.builder._world_map._proj = self.builder._world_map.proj.clone(
                cam_pose=current_pose,
                cam_height=current_height,
                cam_pitch=current_pitch,
            )
            self.builder._world_map._topdown_map[idx] = self.builder.proj.fill_value
            self.builder._world_map._height_map[idx] = float("-inf")
            self.builder._world_map._mask[idx] = False
        return self.builder._world_map

    def update(
        self,
        depth: Tensor,
        values: Tensor,
        cam_pose: Tensor,
        valid: Tensor | None = None,
        cam_pitch: Tensor | None = None,
        cam_height: Tensor | None = None,
    ) -> TopdownMap:
        local_map = self.builder.step(
            depth,
            values,
            valid,
            cam_pose,
            cam_pitch=cam_pitch,
            cam_height=cam_height,
            to_global=False,
        )
        return local_map

    def update_occupancy(
        self,
        occupancy_map: Tensor,
        cam_pose: Tensor,
        cam_pitch: Tensor | None = None,
        cam_height: Tensor | None = None,
        unexplored: int = Occupancy.unexplored,
        occupied: int = Occupancy.occupied,
    ) -> TopdownMap:
        """Plot the new top-down map with occupancy map and merge it to the world map. The camera is assumed to be in the center of the occupancy map.

        Args:
        occupancy_map (Tensor): new 2/3/4D occupancy map. It should be in (h, w),
            (1, h, w), (b, 1, h, w), np.int8. It should have values -1 (unknown), 0 (free), 1 (occupied).
        cam_pose (Tensor): camera pose of the new occupancy map. This is
            used to match the coordinates of the new top-down map with the world
            map. Defaults to None.
        merge (bool, optional): whether to merge the top-down map into the world map. Defaults to True.
        Returns:
        kwargs (Dict[str, Any]): options for the local map properties, you can change properties like the occupancy map resolution here if your mapping is in a different resolution compared to global map resolution. Look at  See `orth_project()` for the full argument list.
        TopdownMap: topdown map
        """
        # plot top-down map
        mask = occupancy_map != unexplored
        height_map = torch.zeros_like(occupancy_map, device=occupancy_map.device)
        height_map[occupancy_map == occupied] = 1
        height_map[occupancy_map == unexplored] = float("-inf")
        width = occupancy_map.shape[-1]
        height = occupancy_map.shape[-2]
        # camera is at the center of the occupancy map
        width_offset = width // 2
        height_offset = height // 2

        local_map = TopdownMap(
            topdown_map=occupancy_map,
            mask=mask,
            height_map=height_map,
            map_projector=self.builder.proj.clone(
                cam_pose=cam_pose,
                width_offset=width_offset,
                height_offset=height_offset,
                map_width=width,
                map_height=height,
                to_global=False,
                cam_height=cam_height,
                cam_pitch=cam_pitch,
            ),
            is_height_map=False,
        )
        self.builder.merge(local_map, keep_pose=False)
        return local_map

    @property
    def world_map(self) -> TopdownMap:
        return self.builder._world_map

    def ego_map(self, width: int, height: int) -> TopdownMap:
        b = self.builder._world_map.proj.cam_pose.shape[0]
        cam_pose = torch.zeros((b, 1, 3), dtype=torch.float32)
        global_cam_pos = self.world_map.get_coords(
            points=cam_pose, is_global=False
        )  # (b, 1, 2)
        crop_size = math.ceil(max(width, height) * math.sqrt(2))
        # Select area around the camera in the global map where rotation won't destroy any data of the occupancy map we want to select. We select this small area to make the rotation efficient.
        smaller_world_map = self.world_map.select(
            global_cam_pos, int(crop_size), int(crop_size)
        )
        # Rotate the local map to align with the camera direction to get the ego-centric map
        large_ego_map = dmap.fuse_topdown_maps(
            smaller_world_map,
            map_projector=smaller_world_map.proj.clone(to_global=False),
        )
        ego_cam_pos = large_ego_map.get_camera()
        ego_map = large_ego_map.select(ego_cam_pos, width, height)
        return ego_map

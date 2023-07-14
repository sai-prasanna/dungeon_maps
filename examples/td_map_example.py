# --- built in ---
import math
import random

import cv2

# --- 3rd party ---
import numpy as np
import pyastar
import skimage
import torch

# --- my module ---
import dungeon_maps as dmap

# import simulators, so that one can use dmap.sim.make()
# to create simulators.
import dungeon_maps.sim
import vis
from dungeon_maps.topdown_mapper import TopDownMapper

# Some constants
WIDTH, HEIGHT = 600, 600
HFOV = math.radians(70)
CAM_PITCH = math.radians(-10)
CAM_HEIGHT = 0.88  # meter
MIN_DEPTH = 0.1  # meter
MAX_DEPTH = 10.0  # meter
TRUNC_MAX_DEPTH = 5.0
NUM_CLASSES = 5

MAP_WIDTH = 600
MAP_HEIGHT = 600


def denormalize(depth_map):
    """Denormalize depth map, from [0, 1] to [MIN_DEPTH, MAX_DEPTH]"""
    return depth_map * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH


def render_scene(rgb, depth, seg, world_map, local_map, cam_pose):
    bgr_image = rgb[..., ::-1].astype(np.uint8)  # (h, w, 3)
    depth_image = np.concatenate((depth,) * 3, axis=-1)  # (h, w, 3)
    depth_image = (depth_image * 255.0).astype(np.uint8)
    # seg_image = vis.draw_segmentation(seg)  # (h, w, 3)
    # scene = np.concatenate((bgr_image, depth_image, seg_image), axis=1)
    scene = np.concatenate((bgr_image, depth_image), axis=1)

    # Plot occlusion map
    local_occ_map = vis.draw_map(local_map)
    cam_pos = world_map.get_camera()
    crop_map = world_map.select(cam_pos, local_occ_map.shape[1], local_occ_map.shape[0])
    crop_occ_map = vis.draw_map(crop_map)
    # Concat occlution maps
    local_occ_map = np.pad(
        local_occ_map, ((0, 0), (25, 25), (0, 0)), mode="constant", constant_values=0
    )
    crop_occ_map = np.pad(
        crop_occ_map, ((0, 0), (25, 25), (0, 0)), mode="constant", constant_values=0
    )
    occ_map = np.concatenate((local_occ_map, crop_occ_map), axis=1)
    # padding to same size
    pad_num = np.abs(occ_map.shape[1] - scene.shape[1])
    left_pad = pad_num // 2
    right_pad = pad_num - left_pad
    if scene.shape[1] < occ_map.shape[1]:
        scene = np.pad(
            scene,
            ((0, 0), (left_pad, right_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif scene.shape[1] > occ_map.shape[1]:
        occ_map = np.pad(
            occ_map,
            ((0, 0), (left_pad, right_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    scene = np.concatenate((scene, occ_map), axis=0)
    return scene


def find_frontier(occupancy_map, agent_pos):
    """
    Finds frontiers in an occupancy map around the agent's position.

    Args:
        occupancy_map (ndarray): Occupancy map representing the environment.
        agent_pos (tuple): Position of the agent in the occupancy map (x, y).

    Returns:
        tuple: The goal position (x, y) of the largest cluster, and the frontier indices.
    """
    # Dilate the occupancy map to smoothen the map
    occupancy_map = cv2.dilate(occupancy_map, np.ones((5, 5), np.uint8), iterations=1)

    # Prepare components image for labeling
    components_img = np.ones_like(occupancy_map)
    components_img[occupancy_map == 0.5] = 0.5
    components_img[
        int(agent_pos[1]) - 4 : int(agent_pos[1]) + 4,
        int(agent_pos[0]) - 4 : int(agent_pos[0]) + 4,
    ] = 0.5

    # Label the components in the image
    components_labels = skimage.morphology.label(
        components_img, connectivity=2, background=1, return_num=False
    )
    connected_idx = (
        components_labels == components_labels[int(agent_pos[1]), int(agent_pos[0])]
    )

    # Create a structuring element for morphological operations
    selem = skimage.morphology.disk(1)

    # Find indices for different conditions
    empty_idx = occupancy_map == 0.5
    neighbor_unexp_idx = (
        skimage.filters.rank.minimum((occupancy_map * 255).astype(np.uint8), selem) == 0
    )
    neighbor_occp_idx = (
        skimage.filters.rank.maximum((occupancy_map * 255).astype(np.uint8), selem)
        == 255
    )

    # Get frontier indices and prepare cluster image
    frontier_idx = empty_idx & neighbor_unexp_idx & (~neighbor_occp_idx)
    valid_idx = frontier_idx & connected_idx
    cluster_img = valid_idx.astype(np.uint8)
    labels_cluster = skimage.measure.label(
        cluster_img, connectivity=2, return_num=False
    )

    if cluster_img.sum() != 0:
        # Identify the largest cluster
        counts = np.bincount(labels_cluster.ravel())
        largest_cluster_label = np.argmax(counts[1:]) + 1

        # Prepare the largest cluster image
        largest_cluster_img = np.zeros_like(labels_cluster)
        largest_cluster_img[labels_cluster == largest_cluster_label] = 1

        # Compute the goal position
        final_idx_coords = np.where(largest_cluster_img == 1)
        x_np, y_np = final_idx_coords[0], final_idx_coords[1]
        distances = np.sum(
            (np.subtract.outer(x_np, x_np) ** 2 + np.subtract.outer(y_np, y_np) ** 2)
            ** 0.5,
            1,
        )
        frontier_position = np.array(
            [y_np[np.argmin(distances)], x_np[np.argmin(distances)]]
        )
    else:
        frontier_position = None

    return frontier_position, frontier_idx


def run_example():
    env = dmap.sim.make(
        "dungeon",
        width=WIDTH,
        height=HEIGHT,
        hfov=HFOV,
        cam_pitch=CAM_PITCH,
        cam_height=CAM_HEIGHT,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
    )
    builder = TopDownMapper.create_semantic_mapper(
        camera_hw=(HEIGHT, WIDTH),
        hfov=HFOV,
        trunc_depth_max=5.0,
        trunc_depth_min=0.1,
        map_res=0.05,
        cam_pose=torch.tensor([[0.0, 0.0, 0.0]]),
        cam_pitch=torch.tensor([CAM_PITCH]),
        cam_height=torch.tensor([CAM_HEIGHT]),
    )

    # Reset simulator and map builder
    observations = env.reset()
    builder.reset(idx=[0], cam_pose=observations["pose_gt"])
    while True:
        # RGB image (h, w, 3), np.uint8
        rgb = observations["rgb"]
        # Depth image (h, w, 1), np.float32
        depth = observations["depth"]
        # Segmentation image (h, w, 1), np.int64
        # seg = observations["segmentation"]
        # Ground truth camera pose [x, z, yaw] in world coordinate
        cam_pose = observations["pose_gt"].astype(np.float32)
        # Denormalized depth map to [MIN_DEPTH, MAX_DEPTH]
        depth_map = np.transpose(denormalize(depth), (2, 0, 1))  # (1, h, w)
        # Project height map from depth map
        # One can enable GPU acceleration by converting depth map to
        # torch.Tensor and placing it on cuda devices. For example:
        #   depth_map = torch.tensor(depth_map, device='cuda')
        # other variables will be converted to torch.Tensor automatically.
        depth_map = torch.tensor(depth_map, device="cuda").unsqueeze(0)
        # seg_map = torch.tensor(seg, device="cuda")
        # Convert to one-hot encoding
        # seg_map = seg_map.squeeze(dim=-1).unsqueeze(0).unsqueeze(0).to(torch.float32)  # (1, 1, h, w)
        local_map = builder.update(depth_map, None, torch.tensor(cam_pose).unsqueeze(0))

        # render scene
        scene = render_scene(
            rgb, depth, None, builder.builder.world_map, local_map, cam_pose
        )
        cv2.imshow("Object map", scene)

        wmap = builder.builder.world_map
        wmap_color = vis.draw_map(wmap)

        cam_pos = wmap.get_camera()
        x, y = cam_pos[:, 0], cam_pos[:, 1]
        x[x > wmap.topdown_map.shape[-1] - 1] = wmap.topdown_map.shape[-1] - 1
        y[y > wmap.topdown_map.shape[-2] - 1] = wmap.topdown_map.shape[-2] - 1
        cam_coord = torch.stack([x, y], dim=1).long()
        occupancy = torch.zeros_like(wmap.topdown_map)
        occupancy[wmap.topdown_map <= 0.2] = 0.5
        occupancy[wmap.topdown_map > 0.2] = 1.0
        occupancy[~wmap.mask] = 0

        ocp_map = occupancy[0][0].cpu().numpy()
        frontier_point, frontier_idx = find_frontier(
            ocp_map, cam_coord[0].cpu().numpy()
        )
        wmap_color[frontier_idx] = [255, 0, 0]

        ego_map = builder.ego_map(width=250, height=250)
        ego_map_color = vis.draw_map(ego_map)

        def get_border_point(img_shape, point):
            img_center = np.array([img_shape[0] / 2, img_shape[1] / 2])
            point = np.array(point)

            # Check if point is within the borders
            if 0 <= point[0] < img_shape[0] and 0 <= point[1] < img_shape[1]:
                return point

            direction = point - img_center

            if direction[0] == 0:  # Point is directly above or below the center
                return (
                    (point[0], 0) if direction[1] < 0 else (point[0], img_shape[1] - 1)
                )

            slope = direction[1] / direction[0]

            if direction[0] > 0:  # Point is on right side
                x_border = img_shape[0] - 1
            else:  # Point is on left side
                x_border = 0

            y_border = slope * (x_border - img_center[0]) + img_center[1]

            if y_border < 0:  # Point is above (in the image coordinate system)
                y_border = 0
                x_border = (y_border - img_center[1]) / slope + img_center[0]
            elif (
                y_border >= img_shape[1]
            ):  # Point is below (in the image coordinate system)
                y_border = img_shape[1] - 1
                x_border = (y_border - img_center[1]) / slope + img_center[0]

            return round(x_border), round(y_border)

        if frontier_point is not None:
            random_goal = random.choice(np.argwhere(ocp_map == 0.5))

            agent_pos = wmap.get_camera().cpu().numpy()[0]

            traversable_map = np.zeros_like(ocp_map, dtype=np.uint8)
            traversable_map[ocp_map == 0.5] = 1
            path = plan_path(
                agent_pos[::-1], random_goal, traversable_map, 0.05, 0.1, 0.2
            )
            if path is not None:
                wmap_color[path[:, 0], path[:, 1]] = [0, 255, 0]

            wmap_color[
                frontier_point[1] : frontier_point[1] + 5,
                frontier_point[0] : frontier_point[0] + 5,
                :,
            ] = [0, 0, 255]
            wmap_color[frontier_idx] = [255, 0, 0]

            frontiers_map_xz = np.array([frontier_point[0], frontier_point[1]])

            frontiers_global_x, frontiers_global_z = wmap.proj.map_dequantize(
                frontiers_map_xz[0], frontiers_map_xz[1]
            )
            frontiers_global_xyz = np.array(
                [frontiers_global_x.item(), 0, frontiers_global_z.item()]
            )
            ego_map_frontier_xz = ego_map.get_coords(
                frontiers_global_xyz, is_global=True
            )[0][0]
            ego_map_frontier_xz = get_border_point(
                (ego_map_color.shape[1], ego_map_color.shape[0]), ego_map_frontier_xz
            )
            ego_map_color[
                ego_map_frontier_xz[1] : ego_map_frontier_xz[1] + 5,
                ego_map_frontier_xz[0] : ego_map_frontier_xz[0] + 5,
                :,
            ] = [0, 0, 255]

        cv2.imshow("w map", wmap_color)

        cv2.imshow("egomap", ego_map_color)

        print("global pose", cam_pose)
        print("map coordinate", cam_pos)

        # Taking actions via keyboard inputs
        key = cv2.waitKey(0)
        if key == ord("w"):
            action = env.FORWARD
        elif key == ord("a"):
            action = env.LEFT
        elif key == ord("s"):
            action = env.BACKWARD
        elif key == ord("d"):
            action = env.RIGHT
        elif key == ord("q"):
            print("Quit")
            exit()
        else:
            action = env.NONE
        observations = env.step(action)


def plan_path(
    start: np.ndarray,
    end: np.ndarray,
    traversable_map: np.ndarray,
    map_resolution: float,
    plan_resolution: float | None = None,
    erode_radius: float | None = None,
):
    plan_resolution = plan_resolution or map_resolution

    # Ensure start is within bounds.
    if (
        start[0] < 0
        or start[0] >= traversable_map.shape[0]
        or start[1] < 0
        or start[1] >= traversable_map.shape[1]
    ) or (
        end[0] < 0
        or end[0] >= traversable_map.shape[0]
        or end[1] < 0
        or end[1] >= traversable_map.shape[1]
    ):
        return None

    # Calculate the scaling factor for resizing the map
    scale_factor = plan_resolution / map_resolution
    new_size = (
        int(traversable_map.shape[1] / scale_factor),
        int(traversable_map.shape[0] / scale_factor),
    )

    # Resize the traversable map
    traversable_map_small = cv2.resize(
        traversable_map, new_size, interpolation=cv2.INTER_LINEAR
    )
    # Scale start and end to match the graph
    start_scaled = tuple((start / scale_factor).astype(int))
    end_scaled = tuple((end / scale_factor).astype(int))

    if erode_radius is not None:
        # Calculate the robot radius in pixels at the planning resolution
        robot_radius_px = int(erode_radius / plan_resolution)

        # Create a structuring element for dilation
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * robot_radius_px + 1, 2 * robot_radius_px + 1)
        )

        # Erode the free space in the map
        traversable_map_small = cv2.erode(
            traversable_map_small, se, borderType=cv2.BORDER_CONSTANT, borderValue=1
        )
        traversable_map_small[start_scaled[0], start_scaled[1]] = traversable_map[
            start[0], start[1]
        ]
        traversable_map_small[end_scaled[0], end_scaled[1]] = traversable_map[
            end[0], end[1]
        ]

    # Plan path with A*
    try:
        weight_map = traversable_map_small.astype(np.float32)
        weight_map[weight_map == 0] = float("inf")
        path = pyastar.astar_path(
            weight_map, start_scaled, end_scaled, costfn="l2", allow_diagonal=True
        )
    except ValueError as e:
        path = None
    # If a path was found, scale it back up to match the original map resolution
    if path is not None:
        path = (np.array(path) * scale_factor).astype(int)
    return path


# def plan_path(start: np.ndarray, end: np.ndarray, traversable_map: np.ndarray, map_resolution: float, plan_resolution: float, robot_radius: float):
#     # Calculate the scaling factor for resizing the map
#     scale_factor = plan_resolution / map_resolution
#     new_size = (int(traversable_map.shape[1] / scale_factor), int(traversable_map.shape[0] / scale_factor))

#     # Resize the traversable map
#     traversable_map_small = cv2.resize(traversable_map, new_size, interpolation=cv2.INTER_LINEAR)

#     # Calculate the robot radius in pixels at the planning resolution
#     robot_radius_px = int(robot_radius / plan_resolution)

#     # Create a structuring element for dilation
#     se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * robot_radius_px + 1, 2 * robot_radius_px + 1))

#     # Dilate the obstacles in the map
#     # Invert the map, dilate the obstacles in the map, then invert it back
#     traversable_map_small = 1 - traversable_map_small
#     traversable_map_small = cv2.dilate(traversable_map_small, se, borderType=cv2.BORDER_CONSTANT, borderValue=1)
#     traversable_map_small = 1 - traversable_map_small

#     # Build the graph
#     map_height, map_width = traversable_map_small.shape
#     g = nx.Graph()
#     for i in range(map_height):
#         for j in range(map_width):
#             if traversable_map_small[i, j] == 0:
#                 continue
#             g.add_node((i, j))
#             # 8-connected graph
#             neighbors = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
#             for n in neighbors:
#                 if (
#                     0 <= n[0] < map_height
#                     and 0 <= n[1] < map_width
#                     and traversable_map_small[n[0], n[1]] > 0
#                 ):
#                     g.add_edge(n, (i, j), weight=l2_distance(n, (i, j)))

#     # Scale start and end to match the graph
#     start_scaled = tuple((start / scale_factor).astype(int))
#     end_scaled = tuple((end / scale_factor).astype(int))

#     # Plan path with A*
#     path = None
#     if start_scaled in g.nodes and end_scaled in g.nodes:
#         try:
#             path = nx.astar_path(g, start_scaled, end_scaled, heuristic=l2_distance)
#         except nx.NetworkXNoPath as e:
#             print("No path found between start and end")
#             path = None

#     # If a path was found, scale it back up to match the original map resolution
#     if path is not None:
#         path = (np.array(path) * scale_factor).astype(int)

#     return path

if __name__ == "__main__":
    run_example()

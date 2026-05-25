# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sapien
import torch
from typing import Any, Union

from mani_skill.agents.robots import Panda, SO100, Fetch, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("DistributeObject-v1", max_episode_steps=120)
class DistributeObjectEnv(BaseEnv):
    """Move several overlapped cubes to evenly arranged target regions on the tabletop.

    Target regions are arranged as:
    - 2 objects: on a line segment
    - 3+ objects: at vertices of a regular polygon
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "xarm6_robotiq", "so100", "widowxai"]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]

    num_objects = 1
    # 每个物块在构建时随机采样一个尺寸，形成"大小不同"的组合
    obj_half_size_range = (0.02, 0.03)

    # 初始堆叠中心区域（尽量重合）
    spawn_center_region = [-0.08, 0.08, -0.08, 0.08]  # [min_x, max_x, min_y, max_y]
    stack_xy_jitter = 0.005

    # 目标区域参数
    goal_region_center = [0.0, 0.0]  # 目标区域的整体中心
    goal_region_radius = 0.25  # 正多边形外接圆半径（或直线半长）
    goal_thresh = 0.02  # 物体到目标点的距离阈值，小于此值算到达

    static_speed_thresh = 0.01

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.05, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
            self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
            self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
            self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
            self.human_cam_target_pos = cfg["human_cam_target_pos"]
        else:
            self.sensor_cam_eye_pos = [0.3, 0, 0.4]
            self.sensor_cam_target_pos = [0, 0, 0.1]
            self.human_cam_eye_pos = [0.7, 0.5, 0.7]
            self.human_cam_target_pos = [0, 0, 0.1]

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _generate_goal_positions_pattern(self, n: int) -> np.ndarray:
        """Generate n goal positions arranged in a regular pattern.

        All goals lie on the circumcircle of radius `goal_region_radius`
        centered at `goal_region_center`, evenly spaced starting from the
        "top" (angle = pi/2). This guarantees that when objects spawn at the
        center of that circle (see `_initialize_episode`), the initial
        obj-to-goal distance is always exactly `goal_region_radius` and the
        environment never accidentally `success` at t=0.

        Works uniformly for any n >= 1:
          - n=1 : a single point at the top of the circle.
          - n=2 : two diametrically opposite points (top & bottom).
          - n>=3: vertices of a regular n-polygon inscribed in the circle.

        Returns: np.ndarray of shape (n, 2) with (x, y) positions.
        """
        cx, cy = self.goal_region_center
        r = self.goal_region_radius
        positions = np.zeros((n, 2))
        for i in range(n):
            angle = 2 * np.pi * i / n + np.pi / 2  # start from top
            positions[i] = [cx + r * np.cos(angle), cy + r * np.sin(angle)]
        return positions

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 为每个物块固定采样一个尺寸（不同物块大小不同）
        self.obj_half_sizes = np.random.uniform(
            low=self.obj_half_size_range[0],
            high=self.obj_half_size_range[1],
            size=(self.num_objects,),
        ).tolist()

        # 创建待操作物体
        self.objs = []
        for i, half_size in enumerate(self.obj_half_sizes):
            color = [float(0.2 + 0.6 * (i / max(self.num_objects - 1, 1))), 0.2, 1.0, 1.0]
            obj = actors.build_cube(
                self.scene,
                half_size=float(half_size),
                color=color,
                name=f"distribute_object_{i}",
                initial_pose=sapien.Pose(p=[0, 0, float(half_size)]),
            )
            self.objs.append(obj)

        # 生成目标区域位置（正多边形顶点或直线排列）
        self.goal_positions_np = self._generate_goal_positions_pattern(self.num_objects)

        # 创建目标区域可视化标记（绿色半透明球体，kinematic，无碰撞）
        self.goal_sites = []
        for i in range(self.num_objects):
            gx, gy = self.goal_positions_np[i]
            goal_site = actors.build_sphere(
                self.scene,
                radius=self.goal_thresh,
                color=[0, 1, 0, 0.5],  # Green, semi-transparent
                name=f"goal_site_{i}",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=[float(gx), float(gy), 0.01]),
            )
            self.goal_sites.append(goal_site)
            self._hidden_objects.append(goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- 物体初始位置 ---
            # 固定放在目标区域外接圆的圆心，这样初始时每个物体到任意目标的
            # 距离恰好等于 goal_region_radius（n=1 时 ≈ 0.3 m），远大于
            # goal_thresh（默认 0.02 m），可彻底避免"初始就 success"导致
            # episode 在第 1 步就结束、污染 PPO 的 returns 估计。
            cx, cy = self.goal_region_center
            center_xy = torch.zeros((b, 2), device=self.device)
            center_xy[:, 0] = cx
            center_xy[:, 1] = cy

            # 通过共享中心 + 微小抖动 + 竖直堆叠，制造"尽可能重合"的起始状态
            z_cursor = torch.zeros((b,), device=self.device)
            for i, obj in enumerate(self.objs):
                half_size = float(self.obj_half_sizes[i])
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, :2] = center_xy + (torch.rand((b, 2), device=self.device) - 0.5) * 2 * self.stack_xy_jitter
                z_cursor = z_cursor + half_size
                xyz[:, 2] = z_cursor 
                z_cursor = z_cursor + half_size * 0.85

                qs = torch.zeros((b, 4), device=self.device)
                angles = torch.rand((b), device=self.device) * 2 * np.pi
                qs[:, 0] = torch.cos(angles / 2)
                qs[:, 3] = torch.sin(angles / 2)

                obj.set_pose(Pose.create_from_pq(xyz, qs))

            # --- 目标区域随机化：只做整体角度偏置，中心保持在
            # `self.goal_region_center` 不动。这样初始 obj-to-goal 距离永远
            # 等于 goal_region_radius，episode 长度可预期；同时角度随机化
            # 仍然保留了任务多样性（每局目标朝向不同）。 ---
            rotation_angle = torch.rand((b,), device=self.device) * 2 * np.pi  # [B]
            cos_a = torch.cos(rotation_angle)  # [B]
            sin_a = torch.sin(rotation_angle)  # [B]

            goal_positions_tensor = torch.tensor(
                self.goal_positions_np, dtype=torch.float32, device=self.device
            )  # [N, 2]

            for i, goal_site in enumerate(self.goal_sites):
                # 基础位置（相对于中心）
                base_x = goal_positions_tensor[i, 0] - self.goal_region_center[0]
                base_y = goal_positions_tensor[i, 1] - self.goal_region_center[1]

                # 围绕固定中心旋转
                rotated_x = cos_a * base_x - sin_a * base_y  # [B]
                rotated_y = sin_a * base_x + cos_a * base_y  # [B]

                # 中心固定不再加偏移
                goal_xyz = torch.zeros((b, 3), device=self.device)
                goal_xyz[:, 0] = rotated_x + self.goal_region_center[0]
                goal_xyz[:, 1] = rotated_y + self.goal_region_center[1]
                goal_xyz[:, 2] = 0.01  # slightly above table

                goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    @property
    def obj_positions(self) -> torch.Tensor:
        """Returns [B, N, 3]"""
        return torch.stack([obj.pose.p for obj in self.objs], dim=1)

    @property
    def goal_positions(self) -> torch.Tensor:
        """Returns [B, N, 3]"""
        return torch.stack([gs.pose.p for gs in self.goal_sites], dim=1)

    @property
    def obj_speeds(self) -> torch.Tensor:
        return torch.stack(
            [torch.linalg.norm(obj.linear_velocity, dim=1) for obj in self.objs], dim=1
        )

    def _compute_obj_goal_assignment(self, obj_pos: torch.Tensor, goal_pos: torch.Tensor):
        """Compute optimal assignment from objects to goals using greedy matching.

        Returns:
            assigned_dists: [B, N] distances from each object to its assigned goal
            assigned_goal_idx: [B, N] the goal index assigned to each object
        """
        b, n = obj_pos.shape[0], obj_pos.shape[1]
        obj_xy = obj_pos[:, :, :2]  # [B, N, 2]
        goal_xy = goal_pos[:, :, :2]  # [B, N, 2]

        # Pairwise distance matrix [B, N_obj, N_goal]
        dist_matrix = torch.linalg.norm(
            obj_xy.unsqueeze(2) - goal_xy.unsqueeze(1), dim=-1
        )

        assigned_dists = torch.zeros((b, n), device=obj_pos.device)
        assigned_goal_idx = torch.zeros((b, n), dtype=torch.long, device=obj_pos.device)

        work_dist = dist_matrix.clone()

        for _ in range(n):
            flat_dist = work_dist.reshape(b, -1)  # [B, N*N]
            min_idx = flat_dist.argmin(dim=1)  # [B]
            obj_idx = min_idx // n
            goal_idx = min_idx % n

            batch_idx = torch.arange(b, device=obj_pos.device)
            # 下面的操作等价于：
            # for bi in range(b):
            #   assigned_dists[bi, obj_idx[bi]] = dist_matrix[bi, obj_idx[bi], goal_idx[bi]]
            assigned_dists[batch_idx, obj_idx] = dist_matrix[batch_idx, obj_idx, goal_idx]
            assigned_goal_idx[batch_idx, obj_idx] = goal_idx

            for bi in range(b):
                oi = obj_idx[bi].item()
                gi = goal_idx[bi].item()
                work_dist[bi, oi, :] = float('inf')
                work_dist[bi, :, gi] = float('inf')

        return assigned_dists, assigned_goal_idx

    def _get_obs_extra(self, info: dict):
        # NOTE: 不要依赖 `info` 里 evaluate() 填充的字段。ManiSkill 在
        # `BaseEnv.__init__` 里会调用一次 `reset(... reconfigure=True)`，
        # 此时第一次 `get_obs(info)` 的 info 还是空 dict（evaluate 尚未调用），
        # 任何 `info["xxx"]` 都会 KeyError。这里所有量都用 self.* 现算。
        goal_positions = self.goal_positions  # [B, N, 3]
        obj_positions = self.obj_positions  # [B, N, 3]
        obj_poses = torch.stack([obj.pose.raw_pose for obj in self.objs], dim=1)
        b = obj_poses.shape[0]
        tcp_p = self.agent.tcp_pose.p  # [B, 3]

        # 相对量：TCP -> 每个物体；每个物体 -> 其最近的目标（贪心匹配的逐对距离）
        tcp_to_objs = obj_positions - tcp_p.unsqueeze(1)  # [B, N, 3]
        obj_to_goal_dists, _ = self._compute_obj_goal_assignment(
            obj_positions, goal_positions
        )  # [B, N]

        # is_grasping 也现算，避免依赖 info
        is_grasping = torch.zeros(b, dtype=torch.float32, device=obj_poses.device)
        for obj in self.objs:
            grasp = self.agent.is_grasping(obj)
            grasp = grasp.squeeze() if grasp.dim() > 1 else grasp
            is_grasping = is_grasping + grasp.float()
        is_grasping = (is_grasping > 0).float().unsqueeze(-1)  # [B, 1]

        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            obj_poses=obj_poses.reshape(b, -1),
            goal_positions=goal_positions.reshape(b, -1),
            tcp_to_objs=tcp_to_objs.reshape(b, -1),  # [B, N*3]
            obj_to_goal_dist=obj_to_goal_dists.reshape(b, -1),  # [B, N]
            is_grasping=is_grasping,  # [B, 1]
        )
        return obs

    def evaluate(self):
        obj_positions = self.obj_positions  # [B, N, 3]
        goal_positions = self.goal_positions  # [B, N, 3]

        # Compute assignment distances and indices
        obj_to_goal_dists, assigned_goal_idx = self._compute_obj_goal_assignment(
            obj_positions, goal_positions
        )  # [B, N], [B, N]

        # Per-object mask: whether each object is already at its goal
        obj_at_goal = obj_to_goal_dists < self.goal_thresh  # [B, N]
        all_at_goal = obj_at_goal.all(dim=1)  # [B]

        # Check if objects are static
        obj_speeds = self.obj_speeds  # [B, N]
        max_obj_speed = obj_speeds.max(dim=1).values
        is_static = max_obj_speed < self.static_speed_thresh

        # Check if the robot is grasping any object (OR over all objects)
        is_grasping = torch.zeros(obj_positions.shape[0], dtype=torch.bool, device=obj_positions.device)
        grasped_obj_idx = torch.zeros_like(obj_at_goal, dtype=torch.bool) # [B, N]
        for i, obj in enumerate(self.objs):
            is_obj_grasped = self.agent.is_grasping(obj) # [B]
            is_grasping = is_grasping | is_obj_grasped
            grasped_obj_idx[:, i] = is_obj_grasped.squeeze() if is_obj_grasped.dim() > 1 else is_obj_grasped

        success = all_at_goal & is_static

        return {
            "success": success, # [B]
            "all_at_goal": all_at_goal, # [B]
            "is_static": is_static, # [B]
            "is_grasping": is_grasping, # [B]
            "obj_at_goal": obj_at_goal, # [B, N]
            "obj_to_goal_dists": obj_to_goal_dists,  # [B, N]
            "assigned_goal_idx": assigned_goal_idx, # [B, N]
            "obj_speeds": obj_speeds, # [B, N]
            "grasped_obj_idx": grasped_obj_idx, # [B, N]
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Dense reward with grasp-conditioned approach and global placement reward.

        - Approach reward (only when NOT grasping):
            1 - tanh(mean distance from TCP to all un-placed objects)
            Encourages the gripper to move towards objects that still need placement.

        - Placement reward (always active):
            1 - tanh(mean distance from all objects to their assigned goals)
            Encourages reducing every object's distance to its target.

        - Disturbance penalty (always active):
            Penalizes moving objects that are already at their goals.

        - Success bonus: large bonus when all objects placed and static.
        """
        obj_positions = self.obj_positions  # [B, N, 3]
        tcp_pos = self.agent.tcp_pose.p  # [B, 3]

        obj_to_goal_dists = info["obj_to_goal_dists"]  # [B, N]
        obj_at_goal = info["obj_at_goal"]  # [B, N] bool
        obj_speeds = info["obj_speeds"]  # [B, N]

        # =====================================================================
        # 1) Approach reward
        #    = 1 - tanh(5 * 3D TCP-to-obj distance)
        #    Use full 3D distance (xyz), not just xy: otherwise TCP can sit
        #    high above the cube and still get a "near" reward, which causes
        #    the policy to learn a degenerate "hover-away" behaviour.
        # =====================================================================
        tcp_to_obj_dist = torch.linalg.norm(
            obj_positions - tcp_pos.unsqueeze(1), dim=-1
        )  # [B, N]

        # Mask out already-placed objects so they don't affect the min
        tcp_to_unplaced = tcp_to_obj_dist.clone()
        tcp_to_unplaced[obj_at_goal] = torch.tensor(float('inf'))  # [B, N]
        tcp_to_unplaced_min = tcp_to_unplaced.min(dim=1).values  # [B]
        approach_reward = 1 - torch.tanh(5.0 * tcp_to_unplaced_min)  # [0, 1]

        # =====================================================================
        # 2) Placement reward (always active):
        #    = 1 - tanh( mean obj-to-goal distance across ALL objects )
        # =====================================================================
        placement_reward_individual = 1 - torch.tanh(5.0 * obj_to_goal_dists)  # [B, N]
        placement_reward = torch.sum(placement_reward_individual, dim=1)

        # =====================================================================
        # 3) Disturbance penalty: penalize non-grasped objects being moved
        #    Any object NOT currently being grasped should stay still.
        # =====================================================================
        # grasped_obj_idx = info["grasped_obj_idx"]  # [B, N] bool
        # non_grasped_speeds = obj_speeds.clone()
        # non_grasped_speeds[grasped_obj_idx] = 0.0  # ignore the grasped object's speed
        # disturbance_penalty = torch.tanh(10.0 * non_grasped_speeds.sum(dim=1))  # [0, 1]

        # =====================================================================
        # 4) Combine
        # =====================================================================
        approach_term = 1.0 * approach_reward
        placement_term = 1.0 * placement_reward
        # disturbance_term = -0.3 * disturbance_penalty
        success_bonus = torch.zeros_like(approach_term)
        success_bonus[info["success"]] = 3.0
        grasping_bonus = torch.zeros_like(approach_term)
        grasping_bonus[info["is_grasping"]] = 1.0

        reward = approach_term + placement_term + success_bonus + grasping_bonus # + disturbance_term 

        # Expose per-component rewards through `info` so that the env wrapper
        # (`maniskill_env.MS3VecEnv._record_metrics`) can accumulate them into
        # `info["episode"]` and the runner can log them to tensorboard as
        # `env/reward_<component>`.
        info["reward_approach"] = approach_term.detach()
        info["reward_placement"] = placement_term.detach()
        # info["reward_disturbance"] = disturbance_term.detach()
        info["reward_success_bonus"] = success_bonus.detach()
        info["reward_total"] = reward.detach()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Normalized dense reward."""
        reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        normalized_reward = reward / 4.0
        return normalized_reward    
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


@register_env("MoveObject-v1", max_episode_steps=100)
class MoveObjectEnv(BaseEnv):
    """
    **Task Description:**
    A simple pushing task where the robot needs to push a target object (red cube) 
    from its initial position to a designated goal region on the table using contact forces.
    This task focuses on non-prehensile manipulation skills.
    
    **Randomizations:**
    - The object's initial xy position is randomized on the table in the region [-0.1, 0.1] x [-0.15, -0.05]
    - The goal region's xy position is randomized in the region [-0.1, 0.1] x [0.05, 0.15]
    - The object's z-axis rotation is randomized to a random angle
    
    **Success Conditions:**
    - The object center is within `goal_thresh` (default 0.03m) euclidean distance of the goal position
    - The robot is static (q velocity < 0.2) to ensure stable final state
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "xarm6_robotiq", "so100", "widowxai"]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]
    goal_thresh = 0.03
    obj_half_size = 0.02
    obj_spawn_region = [-0.25, 0.2, -0.4, 0.4]  # [min_x, max_x, min_y, max_y] for object spawn
    goal_spawn_region = [-0.25, 0.2, -0.4, 0.4]   # [min_x, max_x, min_y, max_y] for goal spawn

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.05, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
            self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
            self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
            self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
            self.human_cam_target_pos = cfg["human_cam_target_pos"]
        else:
            # Default camera configs for panda
            self.sensor_cam_eye_pos = [0.3, 0, 0.4]
            self.sensor_cam_target_pos = [0, 0, 0.1]
            self.human_cam_eye_pos = [0.7, 0.5, 0.7]
            self.human_cam_target_pos = [0, 0, 0.1]
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
        # try:
        #     rb = self.table_scene.table._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
        #     aabb = rb.compute_global_aabb_tight()  # (2,3): [[minx,miny,minz],[maxx,maxy,maxz]]
        # except Exception:
        #     # fallback: use collision box params from TableSceneBuilder
        #     c = np.array([-0.12, 0.0, -0.9196429 / 2])
        #     h = np.array([2.418 / 2, 1.209 / 2, 0.9196429 / 2])
        #     aabb = np.stack([c - h, c + h], axis=0)

        # print(f"[table AABB] min={aabb[0]}, max={aabb[1]}")
        # breakpoint()
        # # auto spawn region from AABB (safe margin)
        # margin = float(self.obj_half_size + 0.01)
        # self.obj_spawn_region = [aabb[0,0] + margin, aabb[1,0] - margin,
        #                         aabb[0,1] + margin, aabb[1,1] - margin]
        # self.goal_spawn_region = self.obj_spawn_region.copy()
        # print(f"[spawn] {self.obj_spawn_region}")
        # Create the object to be pushed (red cube)
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.obj_half_size,
            color=[1, 0, 0, 1],  # Red
            name="push_object",
            initial_pose=sapien.Pose(p=[0, 0, self.obj_half_size]),
        )
        
        # Create goal marker (green sphere)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],  # Green
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Randomize object position in the back region of the table
            obj_xyz = torch.zeros((b, 3))
            obj_xyz[:, 0] = (
                torch.rand((b)) * (self.obj_spawn_region[1] - self.obj_spawn_region[0]) 
                + self.obj_spawn_region[0]
            )
            obj_xyz[:, 1] = (
                torch.rand((b)) * (self.obj_spawn_region[3] - self.obj_spawn_region[2]) 
                + self.obj_spawn_region[2]
            )
            obj_xyz[:, 2] = self.obj_half_size
            
            # Randomize object rotation around z-axis only
            qs = torch.zeros((b, 4))
            angles = torch.rand((b)) * 2 * np.pi
            qs[:, 0] = torch.cos(angles / 2)  # w
            qs[:, 3] = torch.sin(angles / 2)  # z
            qs[:, 1:3] = 0  # x, y = 0 for z-axis rotation only
            
            self.obj.set_pose(Pose.create_from_pq(obj_xyz, qs))
            
            # Randomize goal position in the front region of the table
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = (
                torch.rand((b)) * (self.goal_spawn_region[1] - self.goal_spawn_region[0]) 
                + self.goal_spawn_region[0]
            )
            goal_xyz[:, 1] = (
                torch.rand((b)) * (self.goal_spawn_region[3] - self.goal_spawn_region[2]) 
                + self.goal_spawn_region[2]
            )
            goal_xyz[:, 2] = self.obj_half_size  # Same height as object
            
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            obj_pose=self.obj.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp_pose.p,
            obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
        )
        return obs

    def evaluate(self):
        # Check if object is close enough to goal
        obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.obj.pose.p, axis=1)
        is_obj_at_goal = obj_to_goal_dist <= self.goal_thresh
        
        # Check if object is static (q velocity < 0.05)
        obj_velocity = self.obj.linear_velocity
        is_obj_static = torch.linalg.norm(obj_velocity, axis=1) < 0.01
        
        # Success requires both conditions
        success = is_obj_at_goal & is_obj_static
        
        return {
            "success": success,
            "is_obj_at_goal": is_obj_at_goal,
            "is_obj_static": is_obj_static,
            "obj_to_goal_dist": obj_to_goal_dist,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # breakpoint()
        # num_envs = len(info["elapsed_steps"])
        # Distance from TCP to object (encourage approaching)
        not_success_yet = ~info["success"]
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        approach_raw = 1 - torch.tanh(5 * tcp_to_obj_dist)
        approach_reward = approach_raw * not_success_yet
        # Distance from object to goal (main objective)
        obj_to_goal_dist = info["obj_to_goal_dist"]
        push_reward = (1 - torch.tanh(5 * obj_to_goal_dist))*3
        
        # # Contact reward (encourage touching the object)
        # # SAPIEN 3 的 GPU 物理后端不支持 get_contacts() 逐帧查询接触信息，该 API 仅在 CPU 模式下可用。
        # contact_info = self.scene.get_contacts()
        # has_contact = False
        # for contact in contact_info:
        #     if contact.actor0.name == "push_object" or contact.actor1.name == "push_object":
        #         has_contact = True
        #         break
        # contact_reward = float(has_contact) * 0.5
        
        # Static reward when object is near goal (encourage stopping)
        obj_speed = torch.linalg.norm(self.obj.linear_velocity, axis=1)
        static_reward = info["is_obj_static"].float()*info["is_obj_at_goal"].float()*(1-torch.tanh(5 * obj_speed))
        
        # Combine rewards
        reward = approach_reward + push_reward + static_reward
        
        # Bonus for success
        reward[info["success"]] += 3
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Maximum possible reward: approach(1) + push(3)  + static(0.2) + success(3) = 7.2
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 7.2
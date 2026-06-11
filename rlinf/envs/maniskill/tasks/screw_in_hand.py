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

"""In-hand screw rotation task.

Re-implementation (in ManiSkill / SAPIEN) of the simulation task from
``dexscrew`` (XHand-Hora screwdriver / nut-bolt rotation).
The XHand is replaced by ManiSkill's built-in ``AllegroHandRightTouch``;
the screw asset is the original dexscrew URDF/STL collection that has
been copied into ``rlinf/envs/maniskill/assets/screw_in_hand``.

The task spawns a screw (articulated body: ``base`` --fixed-- ``bolt/shaft``
--revolute-- ``nut/handle``) standing on the table.  The hand is initialized
above it with a closing-grip pose.  Reward encourages spinning the nut joint
about its axis (+z by URDF convention) while penalising work / torque / pose
drift and falling.

Available variants (registered):
    * ``ScrewInHand-trinut-v1``
    * ``ScrewInHand-boxnut-v1``
    * ``ScrewInHand-driver-v1``
"""

from pathlib import Path
from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import AllegroHandRightTouch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose, vectorize_pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# --------------------------------------------------------------------------- #
# Asset locations (copied from dexscrew/assets)                               #
# --------------------------------------------------------------------------- #
_ASSET_ROOT = Path(__file__).resolve().parent.parent / "assets" / "screw_in_hand"

SCREW_URDFS = {
    "trinut": _ASSET_ROOT / "screw" / "trinut" / "0003_stripe.urdf",
    "boxnut": _ASSET_ROOT / "screw" / "boxnut" / "0001_stripe.urdf",
    "driver": _ASSET_ROOT / "screw" / "driver" / "0000_stripe.urdf",
}


class ScrewInHandBase(BaseEnv):
    """Base class for in-hand screw rotation.

    Subclasses pick the screw asset (``trinut`` / ``boxnut`` / ``driver``).
    """

    SUPPORTED_ROBOTS = ["allegro_hand_right_touch"]
    agent: Union[AllegroHandRightTouch]

    # Geometry / placement -------------------------------------------------- #
    # Hand hovers above the table; the bolt's base sits on the table top.
    hand_init_height = 0.28
    # Screw URDFs put the heavy base disk centred on z=0; the bolt sticks up
    # to z=0.1 and the nut ring sits around z≈0.1.  Therefore the screw's root
    # pose z should be ~0 so the nut ends up roughly under the palm at
    # ``hand_init_height``.
    screw_root_height = 0.005  # half thickness of the base disk
    # Reward / termination thresholds (mirrors dexscrew defaults).
    reset_dist_threshold = 0.08  # finger ↔ nut distance trigger
    reset_z_drop = 0.10          # if nut falls below this from start → reset

    # Variant selection (subclasses override) ------------------------------- #
    screw_type: str = "trinut"

    def __init__(
        self,
        *args,
        robot_init_qpos_noise: float = 0.02,
        obj_init_pos_noise: float = 0.01,
        num_envs: int = 1,
        reward_scales: dict | None = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.obj_init_pos_noise = obj_init_pos_noise

        # ----- reward weighting (mirrors dexscrew XHandHoraNutBolt yaml) --- #
        default_scales = dict(
            rotate_reward=6.0,
            pose_diff_penalty=-0.5,
            torque_penalty=-0.001,
            work_penalty=-0.001,
            rotate_penalty=-0.3,
            proximity_reward=2.0,
            z_dist_penalty=-1.0,
            fall_penalty=-50.0,
            success_bonus=10.0,
        )
        if reward_scales is not None:
            default_scales.update(reward_scales)
        self._reward_scales = default_scales

        # Reward shaping constants from dexscrew.
        self.angvel_clip_min = -4.0
        self.angvel_clip_max = 4.0
        self.angvel_penalty_thres = 10.0
        # 100 revolutions = 628.32 rad; we declare success at 8π (4 turns)
        # to make Stage-1 PPO tractable while staying faithful to the spirit.
        self.success_threshold = 8.0 * np.pi

        super().__init__(
            *args,
            robot_uids="allegro_hand_right_touch",
            num_envs=num_envs,
            **kwargs,
        )

    # ---------------------------------------------------------------- sim cfg
    @property
    def _default_sim_config(self):
        
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
                found_lost_pairs_capacity=2 ** 26,
            )
        )

    # -------------------------------------------------------------- cameras
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.15, 0.0, 0.55], target=[-0.05, 0.0, self.hand_init_height]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.4, 0.7], [0.0, 0.0, 0.3])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # =====================================================================
    #  Scene loading
    # =====================================================================
    def _load_agent(self, options: dict):
        # Drop the hand "in mid-air" above the screw, palm facing down.
        super()._load_agent(options, sapien.Pose(p=[0, 0, self.hand_init_height]))

    def _load_scene(self, options: dict):
        # Table + ground.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Screw articulation: load from URDF.
        urdf_path = SCREW_URDFS[self.screw_type]
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"Screw URDF not found: {urdf_path}. "
                f"Did you copy dexscrew/assets/screw and meshes into "
                f"rlinf/envs/maniskill/assets/screw_in_hand/ ?"
            )
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True   # base disk is "bolted" to the table
        loader.scale = 1.0
        articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
        assert len(articulation_builders) == 1
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[0, 0, self.screw_root_height])
        self.screw: Articulation = builder.build(name=f"screw_{self.screw_type}")

        # Cache joint / link refs.
        # The articulation has one active revolute joint -> ``nut_joint`` (or
        # ``handle_to_shaft`` for the driver variant).
        active_joints = self.screw.active_joints
        assert len(active_joints) == 1, (
            f"Expected exactly 1 active joint, got {len(active_joints)}"
        )
        self.nut_joint = active_joints[0]

        link_map = {link.name: link for link in self.screw.links}
        # Identify the rotating "nut" link.
        self.nut_link = None
        for cand in ("nut", "handle"):
            if cand in link_map:
                self.nut_link = link_map[cand]
                break
        if self.nut_link is None:
            raise RuntimeError(
                f"Cannot find a nut/handle link in {list(link_map.keys())}"
            )

    # =====================================================================
    #  Episode init
    # =====================================================================
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ---- Hand: closing-grip pose, palm facing down ---------------- #
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = int(dof[0])
            # A reasonably "cupped" allegro pose – we keep it simple and let
            # PPO discover a real grasp; thumb partly closed, fingers slightly
            # curled.  Values are in joint-angle radians.
            init_qpos = torch.zeros((b, dof))
            # Allegro joint order: 0..3 index, 4..7 mid, 8..11 ring, 12..15 thumb
            init_qpos[:, 1] = 0.6       # index curl
            init_qpos[:, 2] = 0.6
            init_qpos[:, 5] = 0.6       # mid curl
            init_qpos[:, 6] = 0.6
            init_qpos[:, 9] = 0.6       # ring curl
            init_qpos[:, 10] = 0.6
            init_qpos[:, 12] = 1.2      # thumb base abduction
            init_qpos[:, 13] = 0.5      # thumb opposition
            init_qpos[:, 14] = 0.5
            init_qpos[:, 15] = 0.5
            init_qpos += (
                torch.randn_like(init_qpos) * self.robot_init_qpos_noise
            )
            self.agent.reset(init_qpos)
            # Palm facing -Z (downward) so fingers curl around the nut from above.
            # Allegro default URDF has palm facing -Z; identity quaternion keeps it.
            # (The "palm_up" keyframe q=[-0.707,0,0.707,0] flips palm to +Z = upward,
            #  which is the opposite of what we want for in-hand manipulation.)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([0.0, 0.0, self.hand_init_height]),
                    q=torch.tensor([0.707, 0.0, 0.707, 0.0]),
                )
            )

            # ---- Screw: small noise on root xy + zero nut angle ----------- #
            noise = torch.randn((b, 3)) * self.obj_init_pos_noise
            noise[:, 2] = 0.0
            root_xyz = torch.zeros((b, 3))
            root_xyz[:, 2] = self.screw_root_height
            root_xyz += noise
            new_pose = Pose.create_from_pq(
                p=root_xyz,
                q=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(b, 4),
            )
            self.screw.set_pose(new_pose)
            zero_qpos = torch.zeros((b, 1))
            self.screw.set_qpos(zero_qpos)
            self.screw.set_qvel(torch.zeros((b, 1)))

            # ---- Per-env state buffers ----------------------------------- #
            self._init_nut_angle = self.screw.qpos[:, 0].clone()
            self._prev_nut_angle = self._init_nut_angle.clone()
            self._init_hand_qpos = self.agent.robot.qpos.clone()
            self._init_nut_z = self.nut_link.pose.p[:, 2].clone()
            self._cum_rotation = torch.zeros((self.num_envs,))
            # Per-env initial buffers may be a subset (env_idx) on resets;
            # fall back to using cur values for the affected envs only.

    # =====================================================================
    #  Observations
    # =====================================================================
    def _get_obs_extra(self, info: dict):
        obs = dict()
        # Asymmetric: under state mode include privileged screw state.
        if self.obs_mode_struct.use_state:
            obs.update(
                screw_root_pose=vectorize_pose(self.screw.pose),
                nut_pose=vectorize_pose(self.nut_link.pose),
                nut_qpos=self.screw.qpos,
                nut_qvel=self.screw.qvel,
                cum_rotation=self._cum_rotation.unsqueeze(-1),
            )
        return obs

    # =====================================================================
    #  Reward / termination
    # =====================================================================
    def evaluate(self, **kwargs) -> dict:
        with torch.device(self.device):
            # 1. nut angular delta (in joint frame, scalar).
            nut_q = self.screw.qpos[:, 0]
            dq = nut_q - self._prev_nut_angle
            self._prev_nut_angle = nut_q.clone()
            # Clip insane single-step jumps (a few rad/step is plenty).
            dq = torch.clip(dq, -np.pi / 5, np.pi / 5)

            # 2. cumulative rotation.
            self._cum_rotation = self._cum_rotation + dq
            success = self._cum_rotation > self.success_threshold

            # 3. nut falling = nut link's z dropped a lot.
            nut_z = self.nut_link.pose.p[:, 2]
            obj_fall = (nut_z < (self._init_nut_z - self.reset_z_drop)).to(
                torch.bool
            )

            # 4. finger ↔ nut distances (use thumb tip and index tip).
            tip_poses = self.agent.tip_poses                          # [b, 4, 7]
            nut_p = self.nut_link.pose.p                              # [b, 3]
            # Allegro ``tip_link_names`` order: thumb, index, mid, ring.
            thumb_pos = tip_poses[:, 0, :3]
            index_pos = tip_poses[:, 1, :3]
            thumb_dist = torch.linalg.norm(thumb_pos - nut_p, dim=-1)
            index_dist = torch.linalg.norm(index_pos - nut_p, dim=-1)
            mean_dist = 0.5 * (thumb_dist + index_dist)

            # 5. proximity reward in [0, 1].
            proximity = torch.clamp(
                1.0 - mean_dist / self.reset_dist_threshold, 0.0, 1.0
            )

            # 6. drop / lose-grip termination – mirror dexscrew check.
            finger_dist_reset = (thumb_dist > self.reset_dist_threshold) | (
                index_dist > self.reset_dist_threshold
            )
            fail = obj_fall | finger_dist_reset

        return dict(
            nut_dq=dq,
            cum_rotation=self._cum_rotation.clone(),
            success=success,
            fail=fail,
            obj_fall=obj_fall,
            thumb_dist=thumb_dist,
            index_dist=index_dist,
            proximity=proximity,
            nut_z=nut_z,
        )

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        with torch.device(self.device):
            s = self._reward_scales

            # rotation reward (clipped joint angular delta scaled to per-second).
            # We approximate angular velocity as dq * control_freq.
            ctrl_freq = float(self.control_freq)
            ang_vel = info["nut_dq"] * ctrl_freq
            rotate_term = torch.clip(
                ang_vel, self.angvel_clip_min, self.angvel_clip_max
            )

            # Penalty when over-spinning (rare; mostly a safety guard).
            rotate_pen = torch.where(
                ang_vel > self.angvel_penalty_thres,
                ang_vel - self.angvel_penalty_thres,
                torch.zeros_like(ang_vel),
            )

            # Pose-diff penalty (excluding thumb DoFs 12..15 mirrors dexscrew).
            qpos = self.agent.robot.qpos
            mask = torch.ones(qpos.shape[-1], device=qpos.device)
            mask[12:] = 0.0
            pose_diff = (((qpos - self._init_hand_qpos) ** 2) * mask).sum(-1)

            # Torque / work penalties – computed from PD controller signals.
            try:
                target = self.agent.controller._target_qpos
                err = target - qpos
                qf = err * self.agent.joint_stiffness - self.agent.robot.qvel * self.agent.joint_damping
                qf = torch.clip(
                    qf,
                    -self.agent.joint_force_limit,
                    self.agent.joint_force_limit,
                )
                torque_pen = (qf ** 2).sum(-1)
                work_pen = (
                    (torch.abs(qf) * torch.abs(self.agent.robot.qvel)).sum(-1)
                ) ** 2
            except Exception:
                torque_pen = torch.zeros(self.num_envs, device=qpos.device)
                work_pen = torch.zeros(self.num_envs, device=qpos.device)

            # z-distance penalty: how much did the nut drop vs. its starting z.
            z_diff = (info["nut_z"] - self._init_nut_z) ** 2

            reward = (
                s["rotate_reward"] * rotate_term
                + s["rotate_penalty"] * rotate_pen
                + s["pose_diff_penalty"] * pose_diff
                + s["torque_penalty"] * torque_pen
                + s["work_penalty"] * work_pen
                + s["proximity_reward"] * info["proximity"]
                + s["z_dist_penalty"] * z_diff
                + s["fall_penalty"] * info["obj_fall"].float()
                + s["success_bonus"] * info["success"].float()
            )
            return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        # Rough upper bound: rotate_reward * angvel_clip_max + success bonus
        max_reward = (
            self._reward_scales["rotate_reward"] * self.angvel_clip_max
            + self._reward_scales["success_bonus"]
            + self._reward_scales["proximity_reward"]
        )
        return self.compute_dense_reward(obs, action, info) / max(max_reward, 1.0)


# --------------------------------------------------------------------------- #
# Registered variants                                                         #
# --------------------------------------------------------------------------- #
@register_env("ScrewInHand-trinut-v1", max_episode_steps=300)
class ScrewInHandTriNut(ScrewInHandBase):
    screw_type = "trinut"


@register_env("ScrewInHand-boxnut-v1", max_episode_steps=300)
class ScrewInHandBoxNut(ScrewInHandBase):
    screw_type = "boxnut"


@register_env("ScrewInHand-driver-v1", max_episode_steps=300)
class ScrewInHandDriver(ScrewInHandBase):
    screw_type = "driver"
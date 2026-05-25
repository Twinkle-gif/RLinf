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

from typing import Optional, OrderedDict, Union

import os
import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import put_info_on_image, tile_images
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

__all__ = ["ManiskillEnv"]


def extract_termination_from_info(info, num_envs, device):
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        else:
            terminated = info["success"].clone()
    else:
        if "fail" in info:
            terminated = info["fail"].clone()
        else:
            terminated = torch.zeros(num_envs, dtype=bool, device=device)
    return terminated


class ManiskillEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        # ----- Potential-based reward shaping (Ng, Harada, Russell 1999) -----
        # When `use_rel_reward=True`, the per-step reward becomes
        #     r'_t = gamma * Phi(s_{t+1}) - Phi(s_t)
        # which is exactly the F function from Ng's Theorem 1 and preserves
        # the optimal policy. `gamma` MUST equal the PPO discount factor
        # (algorithm.gamma); the env yaml passes `gamma: ${algorithm.gamma}`
        # so the two always stay in sync.
        self._shaping_gamma = float(getattr(cfg, "gamma", 0.99))
        self.ignore_terminations = cfg.ignore_terminations
        self.use_full_state = bool(getattr(cfg, "use_full_state", False))
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.env: BaseEnv = gym.make(**env_args) #这里返回的其实不是BaseEnv，而是经过了wrapper包装的，想要拿到BaseEnv需要用self.env.unwrapped   
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )  # [B, ]
        self.record_metrics = record_metrics
        self._is_start = True
        self._init_reset_state_ids()
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        self._show_goal_site_visual()
        if self.record_metrics:
            self._init_metrics()

    @property
    def total_num_group_envs(self):
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        if hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs"):
            return len(self.env.xyz_configs) * len(self.env.quat_configs)
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def instruction(self):
        return self.env.unwrapped.get_language_instruction()

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _show_goal_site_visual(self):
        """Keep ManiSkill goal-site visualization visible for reward-model RGB input."""
        if not hasattr(self.env.unwrapped, "goal_site"):
            return

        goal_site = self.env.unwrapped.goal_site
        if hasattr(self.env.unwrapped, "_hidden_objects"):
            while goal_site in self.env.unwrapped._hidden_objects:
                self.env.unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def _wrap_obs(self, raw_obs, infos=None):
        wrap_obs_mode = getattr(self.cfg, "wrap_obs_mode", "default")
        if wrap_obs_mode == "raw":
            assert infos is not None
            return infos["extracted_obs"]

        if wrap_obs_mode == "simple":
            if self.env.unwrapped.obs_mode == "state":
                return {"states": raw_obs}
            elif self.env.unwrapped.obs_mode == "rgb":
                sensor_data = raw_obs.pop("sensor_data")
                raw_obs.pop("sensor_param")
                if self.use_full_state:
                    state = self._get_full_state_obs()
                else:
                    state = common.flatten_state_dict(
                        raw_obs, use_torch=True, device=self.device
                    )

                main_images = sensor_data["base_camera"]["rgb"]
                sorted_images = OrderedDict(sorted(sensor_data.items()))
                sorted_images.pop("base_camera")
                extra_view_images = (
                    torch.stack([v["rgb"] for v in sorted_images.values()], dim=1)
                    if sorted_images
                    else None
                )
                return {
                    "main_images": main_images,
                    "extra_view_images": extra_view_images,
                    "states": state,
                }

        # Default
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(
            torch.uint8
        )  # [B, H, W, C]
        proprioception: torch.Tensor = self.env.unwrapped.agent.robot.get_qpos().to(
            obs_image.device, dtype=torch.float32
        )
        return {
            "main_images": obs_image,
            "states": proprioception,
            "task_descriptions": self.instruction,
        }

    def _get_full_state_obs(self):
        base_env = self.env.unwrapped
        mode_attr = "_obs_mode" if hasattr(base_env, "_obs_mode") else "obs_mode"
        original_mode = getattr(base_env, mode_attr)
        setattr(base_env, mode_attr, "state")
        try:
            state_obs = base_env.get_obs()
        finally:
            setattr(base_env, mode_attr, original_mode)

        if isinstance(state_obs, dict):
            return common.flatten_state_dict(
                state_obs, use_torch=True, device=self.device
            )
        return state_obs

    def _calc_step_reward(self, reward, info):
        if getattr(self.cfg, "reward_mode", "default") == "raw":
            pass
        elif getattr(self.cfg, "reward_mode", "default") == "only_success":
            reward = info["success"] * 1.0
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
                self.env.unwrapped.device
            )  # [B, ]
            reward += info["is_src_obj_grasped"] * 0.1
            reward += info["consecutive_grasp"] * 0.1
            reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # Potential-based reward shaping (Ng, Harada, Russell 1999):
        #   r'_t = gamma * Phi(s_{t+1}) - Phi(s_t)
        # `reward` here is Phi(s_{t+1}) (post-step); `self.prev_step_reward`
        # stores Phi(s_t) (seeded at reset via `_seed_prev_step_reward`).
        # Policy-invariant in the optimal-policy sense.
        reward_diff = self._shaping_gamma * reward - self.prev_step_reward
        reward_diff = reward_diff / (1-self._shaping_gamma)  # rescale to keep the same reward scale as the original reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        # Per-component reward accumulators. Keys are populated on-the-fly the
        # first time a `reward_*` component appears in `infos`. Any task that
        # writes `info["reward_<name>"] = tensor` in `compute_dense_reward`
        # will get an `env/return_reward_<name>` / `env/mean_reward_<name>`
        # series in tensorboard automatically.
        self.reward_component_returns: dict[str, torch.Tensor] = {}
        # Status of the last `_seed_prev_step_reward` invocation. Surfaced in
        # tensorboard as `env/seed_prev_ok` (1.0 if seeding succeeded last
        # reset, 0.0 otherwise) and `env/seed_prev_r0_mean` (the average r0
        # used as the seed). This lets the user verify from tensorboard alone
        # whether the t=0 reward bias has been removed.
        self.last_seed_status = "uninitialized"
        self.last_seed_r0_mean = 0.0

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                for k in self.reward_component_returns:
                    self.reward_component_returns[k][mask] = 0.0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                for k in self.reward_component_returns:
                    self.reward_component_returns[k][:] = 0.0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]

        # Surface the seed_prev_step_reward status as numeric metrics so the
        # user can verify from tensorboard (`env/seed_prev_ok`,
        # `env/seed_prev_r0_mean`) whether the t=0 reward bias has been removed.
        seed_ok = 1.0 if getattr(self, "last_seed_status", "") == "ok" else 0.0
        episode_info["seed_prev_ok"] = torch.full(
            (self.num_envs,), seed_ok, dtype=torch.float32, device=self.device
        )
        episode_info["seed_prev_r0_mean"] = torch.full(
            (self.num_envs,),
            float(getattr(self, "last_seed_r0_mean", 0.0)),
            dtype=torch.float32,
            device=self.device,
        )

        # Accumulate per-component rewards exposed by `compute_dense_reward`
        # via keys prefixed with "reward_" in `infos`. Both the running return
        # (sum over the episode) and the per-step mean (return / episode_len)
        # are reported, mirroring the behaviour of "return" / "reward".
        for key, value in list(infos.items()):
            if not isinstance(key, str) or not key.startswith("reward_"):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            comp_value = value.detach().to(
                device=self.returns.device, dtype=self.returns.dtype
            )
            if comp_value.shape[0] != self.num_envs:
                continue
            if key not in self.reward_component_returns:
                self.reward_component_returns[key] = torch.zeros_like(self.returns)
            self.reward_component_returns[key] += comp_value
            episode_info[f"return_{key}"] = self.reward_component_returns[key].clone()
            episode_info[f"mean_{key}"] = (
                self.reward_component_returns[key]
                / self.elapsed_steps.clamp(min=1)
            )

        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        if options is None:
            seed = self.seed
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {}
            )
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        self._show_goal_site_visual()
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            env_idx = None
            self._reset_metrics()

        # ---- Seed prev_step_reward with r0 to avoid the t=0 bias in
        # `use_rel_reward` mode. Without this the first step after every reset
        # would emit `r0 - 0 = r0` (the full absolute reward) as if it were a
        # one-step gain, biasing returns/advantages on the first step of each
        # episode. With seeding, the very first `_calc_step_reward` call gets
        # `r1 - r0` — the true incremental progress.
        if self.use_rel_reward:
            self._seed_prev_step_reward(env_idx)
        return extracted_obs, infos

    def _seed_log(self, msg: str):
        """Write a diagnostic message both to stdout (with flush) and to
        `/tmp/rlinf_seed_prev_reward.log`. The latter is bullet-proof against
        Ray's stdout interception."""
        import sys
        try:
            print(f"[ManiskillEnv][seed_prev] {msg}", flush=True)
            sys.stdout.flush()
        except Exception:
            pass
        try:
            with open("/tmp/rlinf_seed_prev_reward.log", "a") as fh:
                fh.write(f"[pid={os.getpid()}] {msg}\n")
        except Exception:
            pass

    def _seed_prev_step_reward(self, env_idx=None):
        """Compute the dense reward at the freshly-reset state and write it
        into `self.prev_step_reward` so that the first incremental reward
        emitted after this reset is a true `r1 - r0` delta (not `r0 - 0`).

        Sets `self.last_seed_status` to one of:
          - 'ok'                       : seeding succeeded
          - 'no_methods'                : task lacks evaluate/compute_dense_reward
          - 'not_tensor'                : compute_dense_reward returned non-Tensor
          - 'shape_mismatch'            : shape of r0 != prev_step_reward
          - 'exception:<type>:<msg>'    : an exception was raised
        This status is then surfaced through every step's `info["episode"]`
        so the user can verify from tensorboard whether seeding worked.
        """
        base_env : BaseEnv = self.env.unwrapped
        if not hasattr(base_env, "compute_dense_reward") or not hasattr(
            base_env, "evaluate"
        ):
            self.last_seed_status = "no_methods"
            if not getattr(self, "_seed_warned_missing", False):
                self._seed_log(
                    "WARNING base_env has no evaluate/compute_dense_reward; "
                    "skipping prev_step_reward seeding"
                )
                self._seed_warned_missing = True
            return

        try:
            info0 = base_env.evaluate()
            r0 = base_env.get_reward(None, None, info0)
        except Exception as e:
            self.last_seed_status = f"exception:{type(e).__name__}:{str(e)[:80]}"
            if not getattr(self, "_seed_warned_exc", False):
                import traceback
                self._seed_log(
                    f"EXCEPTION in evaluate/compute_dense_reward: "
                    f"{type(e).__name__}: {e}"
                )
                self._seed_log(traceback.format_exc())
                self._seed_warned_exc = True
            return

        if not torch.is_tensor(r0):
            self.last_seed_status = "not_tensor"
            self._seed_log(
                f"WARNING compute_dense_reward returned {type(r0).__name__}, "
                f"expected Tensor; skipping seeding"
            )
            return

        r0 = r0.detach().to(
            device=self.prev_step_reward.device, dtype=self.prev_step_reward.dtype
        )
        if r0.shape != self.prev_step_reward.shape:
            self.last_seed_status = "shape_mismatch"
            self._seed_log(
                f"WARNING r0 shape {tuple(r0.shape)} != "
                f"prev_step_reward shape {tuple(self.prev_step_reward.shape)}; "
                f"skipping seeding"
            )
            return

        if env_idx is None:
            self.prev_step_reward[:] = r0
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = r0[mask]
        self.last_seed_status = "ok"
        self.last_seed_r0_mean = float(r0.mean())
        if not getattr(self, "_seed_first_ok_logged", False):
            self._seed_log(
                f"OK first successful seeding, r0 mean={self.last_seed_r0_mean:.4f} "
                f"min={float(r0.min()):.4f} max={float(r0.max()):.4f}"
            )
            self._seed_first_ok_logged = True

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        step_reward = self._calc_step_reward(_reward, infos)

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)
            truncations = truncations.repeat(self.num_envs)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset(options=options)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, info = self.reset()
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )

    # render utils
    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]

        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        return img

    def render(self, info, rew=None):
        if self.video_cfg.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self):
        return self.env.action_space.sample()

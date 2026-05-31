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

import logging

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.mlp_policy.iql_mlp_policy import IQLMLPPolicy
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    iql_config = cfg.get("iql_config", None)
    if iql_config is not None:
        model = IQLMLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
            add_q_head=cfg.get("add_q_head", False),
            q_head_type=cfg.get("q_head_type", "default"),
        )
        model.configure_iql(iql_config)
    else:
        model = MLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
            add_q_head=cfg.get("add_q_head", False),
            q_head_type=cfg.get("q_head_type", "default"),
            critic_obs_dim=cfg.get("critic_obs_dim", None),
            critic_type=cfg.get("critic_type", "mlp"),
            critic_global_dim=cfg.get("critic_global_dim", None),
            critic_per_obj_dim=cfg.get("critic_per_obj_dim", None),
            critic_num_objects=cfg.get("critic_num_objects", None),
            critic_obj_encoder_hidden=cfg.get("critic_obj_encoder_hidden", None),
            critic_final_hidden=cfg.get("critic_final_hidden", None),
            critic_pool_mode=cfg.get("critic_pool_mode", "mean"),
        )
    # Load pretrained weights (partial load supported for fine-tuning with
    # different critic_obs_dim, e.g. transferring from MoveObject to DistributeObject)
    pretrained_path = cfg.get("model_path", None)
    if pretrained_path:
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        # Handle wrapped state dicts (e.g. from FSDP checkpoints)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        # Filter out incompatible keys (e.g. value_head with different dim)
        model_state = model.state_dict()
        compatible_state = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                compatible_state[k] = v
            else:
                skipped_keys.append(k)
        if skipped_keys:
            logger.warning(
                f"Skipped loading {len(skipped_keys)} incompatible keys "
                f"(likely value_head with different critic_obs_dim): {skipped_keys}"
            )
        model.load_state_dict(compatible_state, strict=False)
        logger.info(
            f"Loaded {len(compatible_state)}/{len(state_dict)} parameters from pretrained checkpoint"
        )

    return model

# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-object encoder + pooling critic for set-structured observations.

Designed for multi-object tasks (e.g. DistributeObject) where the critic
observes N objects with identical feature structure. The per-object encoder
is a shared-weight MLP that processes each object independently, then a
pooling operation (sum/mean/max) aggregates the N encodings into a fixed-size
global representation. This global vector is concatenated with scalar/global
features (agent_state, tcp_pose, etc.) and fed into a final output MLP.

Advantages over flat-concatenation:
  - Permutation invariant: object ordering does not matter.
  - Scalable: same parameters work for any N (curriculum learning).
  - Structured: the network can learn per-object patterns before pooling.
"""

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.utils import layer_init


class ObjectSetCritic(nn.Module):
    """Critic head for set-structured observations with per-object encoding.

    Expected input layout (flat tensor):
        [global_features | per_obj_features_flat]

    where per_obj_features_flat = [obj_0_feat, obj_1_feat, ..., obj_{N-1}_feat]
    and each obj_i_feat has `per_obj_dim` elements.

    Architecture:
        per_obj_feat_i -> shared MLP -> h_i   (for each i in 0..N-1)
        h_global = pool(h_0, h_1, ..., h_{N-1})
        output = final_MLP(cat[global_features, h_global])

    Args:
        global_dim: Dimension of the global (non-per-object) features.
        per_obj_dim: Dimension of each per-object feature vector.
        num_objects: Number of objects (used for input parsing).
        obj_encoder_hidden: Hidden sizes for the per-object shared MLP.
        final_hidden: Hidden sizes for the final output MLP.
        pool_mode: Pooling strategy - "mean", "sum", or "max".
        activation: Activation function name ("relu", "tanh", "gelu").
    """

    def __init__(
        self,
        global_dim: int,
        per_obj_dim: int,
        num_objects: int,
        obj_encoder_hidden=(64, 64),
        final_hidden=(256, 256),
        pool_mode: str = "mean",
        activation: str = "tanh",
        output_dim: int = 1,
    ):
        super().__init__()
        self.global_dim = global_dim
        self.per_obj_dim = per_obj_dim
        self.num_objects = num_objects
        self.pool_mode = pool_mode

        # Input dim = global + N * per_obj
        self.input_dim = global_dim + num_objects * per_obj_dim

        # --- Per-object shared encoder ---
        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "gelu":
            act_cls = nn.GELU
        elif activation.lower() == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        obj_layers = []
        in_dim = per_obj_dim
        for h in obj_encoder_hidden:
            obj_layers.append(layer_init(nn.Linear(in_dim, h)))
            obj_layers.append(act_cls())
            in_dim = h
        self.obj_encoder = nn.Sequential(*obj_layers)
        obj_encode_dim = obj_encoder_hidden[-1] if obj_encoder_hidden else per_obj_dim

        # --- Final MLP: global_features + pooled_obj_encoding -> value ---
        final_input_dim = global_dim + obj_encode_dim
        final_layers = []
        in_dim = final_input_dim
        for h in final_hidden:
            final_layers.append(layer_init(nn.Linear(in_dim, h)))
            final_layers.append(act_cls())
            in_dim = h
        final_layers.append(nn.Linear(in_dim, output_dim, bias=False))
        self.final_mlp = nn.Sequential(*final_layers)

        # Initialize final layer
        nn.init.normal_(self.final_mlp[-1].weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, global_dim + N * per_obj_dim] flat observation tensor.

        Returns:
            [B, output_dim] value estimate.
        """
        # Split into global and per-object parts
        global_feat = x[:, :self.global_dim]  # [B, global_dim]
        per_obj_flat = x[:, self.global_dim:]  # [B, N * per_obj_dim]

        # Reshape to [B, N, per_obj_dim]
        b = x.shape[0]
        per_obj = per_obj_flat.reshape(b, self.num_objects, self.per_obj_dim)

        # Encode each object with shared encoder: [B, N, per_obj_dim] -> [B, N, obj_encode_dim]
        # Merge batch and object dims for efficient forward pass
        per_obj_2d = per_obj.reshape(b * self.num_objects, self.per_obj_dim)
        obj_encoded = self.obj_encoder(per_obj_2d)  # [B*N, obj_encode_dim]
        obj_encode_dim = obj_encoded.shape[-1]
        obj_encoded = obj_encoded.reshape(b, self.num_objects, obj_encode_dim)

        # Pool across objects: [B, N, D] -> [B, D]
        if self.pool_mode == "mean":
            pooled = obj_encoded.mean(dim=1)
        elif self.pool_mode == "sum":
            pooled = obj_encoded.sum(dim=1)
        elif self.pool_mode == "max":
            pooled = obj_encoded.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        # Concatenate global features with pooled encoding
        combined = torch.cat([global_feat, pooled], dim=-1)  # [B, global_dim + obj_encode_dim]

        # Final MLP -> value
        return self.final_mlp(combined)
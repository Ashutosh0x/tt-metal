# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtPhiMLP(LightweightModule):
    def __init__(self, device, args, state_dict, layer_num, dtype):
        super().__init__()
        self.device = device
        self.args = args
        
        prefix = f"model.layers.{layer_num}.mlp"
        
        self.w_fc1 = self._load_weight(state_dict, f"{prefix}.fc1.weight", dtype)
        self.b_fc1 = self._load_weight(state_dict, f"{prefix}.fc1.bias", dtype)
        self.w_fc2 = self._load_weight(state_dict, f"{prefix}.fc2.weight", dtype)
        self.b_fc2 = self._load_weight(state_dict, f"{prefix}.fc2.bias", dtype)

    def _load_weight(self, state_dict, name, dtype):
        if state_dict is None or name not in state_dict:
            return None
        weight = state_dict[name]
        if len(weight.shape) == 2:
            weight = weight.T.contiguous()
        return ttnn.from_torch(
            weight,
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x):
        x = ttnn.linear(x, self.w_fc1, bias=self.b_fc1)
        # Use gelu_new (approximate)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.w_fc2, bias=self.b_fc2)
        return x

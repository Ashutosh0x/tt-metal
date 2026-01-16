# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.phi1_5.tt.model_config import PhiConfig, TtPhiArgs
from models.demos.phi1_5.tt.phi_model import TtPhiModel, precompute_rope_phi

__all__ = ["PhiConfig", "TtPhiArgs", "TtPhiModel", "precompute_rope_phi"]

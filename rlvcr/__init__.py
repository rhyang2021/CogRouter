"""
RLVCR (RL with Variational Chain Reasoning) package.

This package implements the RLVCR method for reinforcement learning
with variational chain reasoning optimization.
"""

from .core_rlvcr import (
    compute_rlvcr_outcome_advantage,
    create_level_specific_prompt
)

__all__ = [
    'compute_rlvcr_outcome_advantage',
    'create_level_specific_prompt'
]


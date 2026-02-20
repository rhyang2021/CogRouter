"""
CoPo (RL with Variational Chain Reasoning) package.

This package implements the CoPo method for reinforcement learning
with variational chain reasoning optimization.
"""

from .core_copo import (
    compute_copo_outcome_advantage,
    create_level_specific_prompt
)

__all__ = [
    'compute_copo_outcome_advantage',
    'create_level_specific_prompt'
]


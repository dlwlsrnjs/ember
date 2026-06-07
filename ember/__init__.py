"""EMBER: Direction- and Confidence-Aware LLM Behavior Simulation for
Cold-Start Multi-Task Recommendation.

Public API:
  * :class:`ember.learner.EmberHTLNet` — cold-aware HTLNet learner.
  * :class:`ember.routefuse.RouteFuse` — direction-aware fusion (Eq. 2-3).
  * :class:`ember.calichain.ConfidencePrior`, confidence-weighted losses (Eq. 4-5).
  * :class:`ember.data.EmberHTLDataset`, :func:`ember.data.load_cache`.
"""
from .learner import EmberHTLNet
from .routefuse import RouteFuse
from .calichain import (
    ConfidencePrior,
    confidence_weighted_bce,
    confidence_weighted_mse,
)
from .data import EmberHTLDataset, load_cache

__all__ = [
    "EmberHTLNet",
    "RouteFuse",
    "ConfidencePrior",
    "confidence_weighted_bce",
    "confidence_weighted_mse",
    "EmberHTLDataset",
    "load_cache",
]

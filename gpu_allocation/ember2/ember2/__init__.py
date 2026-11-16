"""Slot denoising for cold-start behavior simulation.

A from-scratch implementation of the method, organized as the paper's three
stages:

  Stage 1  :mod:`ember2.simulate`     direction-specific LoRA simulation ->
                                      hybrid label tuple + per-task confidence
  Stage 2  :mod:`ember2.filterstage`  coupled filter -> top-K candidate pairs
  Stage 3  :mod:`ember2.learner`      RouteFuse + CaliChain cold-aware learner

Everything the LLM does is offline; serving uses only the trained recommender.
"""
from .config import FieldSpec, FilterConfig, LearnerConfig, TaskSpec
from .blocks import (
    InformationFusionUnit,
    LabelEmbeddingUnit,
    SharedFieldEmbedding,
    TaskTower,
)
from .routefuse import RouteFuse
from .calichain import (
    ConfidencePrior,
    confidence_weighted_bce,
    confidence_weighted_mse,
    hybrid_losses,
)
from .learner import EmberLearner
from .gradproc import SharedGradientProcessor
from .dataset import Cache, EvalSplit, TrainDataset
from .rawdata import RawDataset, build_eval_split, make_synthetic
from .cache import build_cache
from .train import evaluate, train

__all__ = [
    "FieldSpec",
    "FilterConfig",
    "LearnerConfig",
    "TaskSpec",
    "SharedFieldEmbedding",
    "TaskTower",
    "LabelEmbeddingUnit",
    "InformationFusionUnit",
    "RouteFuse",
    "ConfidencePrior",
    "confidence_weighted_bce",
    "confidence_weighted_mse",
    "hybrid_losses",
    "EmberLearner",
    "SharedGradientProcessor",
    "Cache",
    "EvalSplit",
    "TrainDataset",
    "RawDataset",
    "make_synthetic",
    "build_eval_split",
    "build_cache",
    "train",
    "evaluate",
]

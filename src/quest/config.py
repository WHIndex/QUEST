from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoreWeights:
    relation: float = 0.50
    endpoint: float = 0.25
    ontology: float = 0.25
    length: float = 0.08


@dataclass(frozen=True)
class SearchConfig:
    direct_budget: int = 32
    path_budget: int = 64
    max_path_length: int = 2
    relation_threshold: float = 0.42
    path_prefix_threshold: float = 0.24
    endpoint_threshold: float = 0.20
    ontology_threshold: float = 0.18
    state_budget: int = 12
    match_budget: int = 8
    recovery_rounds: int = 2
    recovery_passages: int = 8
    runtime_cost_alpha: float = 0.35
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    def __post_init__(self) -> None:
        positive = {
            "direct_budget": self.direct_budget,
            "path_budget": self.path_budget,
            "max_path_length": self.max_path_length,
            "state_budget": self.state_budget,
            "match_budget": self.match_budget,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    search: SearchConfig = field(default_factory=SearchConfig)
    compiler_repairs: int = 2
    context_passages: int = 8
    context_characters: int = 24000
    extraction_batch_size: int = 8

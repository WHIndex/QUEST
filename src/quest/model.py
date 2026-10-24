from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class NodeKind(str, Enum):
    ENTITY = "entity"
    VARIABLE = "variable"
    VALUE = "value"


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class OntologyLabel:
    name: str
    confidence: float


@dataclass(frozen=True)
class EvidenceFact:
    id: str
    subject: str
    relation: str
    object: str
    passage_id: str
    sentence: str
    subject_types: tuple[OntologyLabel, ...] = ()
    relation_types: tuple[OntologyLabel, ...] = ()
    object_types: tuple[OntologyLabel, ...] = ()
    aliases: tuple[str, ...] = ()

    @property
    def triple(self) -> tuple[str, str, str]:
        return self.subject, self.relation, self.object


@dataclass(frozen=True)
class QueryNode:
    id: str
    label: str
    kind: NodeKind
    ontology: tuple[OntologyLabel, ...] = ()
    grounded_id: str | None = None

    @property
    def is_variable(self) -> bool:
        return self.kind == NodeKind.VARIABLE


@dataclass(frozen=True)
class QueryEdge:
    id: str
    source: str
    relation: str
    target: str
    ontology: tuple[OntologyLabel, ...] = ()
    required: bool = True


@dataclass(frozen=True)
class QueryOperator:
    name: str
    arguments: tuple[str, ...]
    value: Any = None


@dataclass(frozen=True)
class QueryGraph:
    nodes: tuple[QueryNode, ...]
    edges: tuple[QueryEdge, ...]
    answer_nodes: tuple[str, ...]
    operators: tuple[QueryOperator, ...] = ()

    def __post_init__(self) -> None:
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("query node identifiers must be unique")
        if len({edge.id for edge in self.edges}) != len(self.edges):
            raise ValueError("query edge identifiers must be unique")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"query edge {edge.id} references an unknown node")
        if not set(self.answer_nodes).issubset(node_ids):
            raise ValueError("answer projection references an unknown node")

    @property
    def node_map(self) -> dict[str, QueryNode]:
        return {node.id: node for node in self.nodes}

    @property
    def edge_map(self) -> dict[str, QueryEdge]:
        return {edge.id: edge for edge in self.edges}

    @property
    def required_edges(self) -> tuple[QueryEdge, ...]:
        return tuple(edge for edge in self.edges if edge.required)

    def variables_needed_after(self, covered: Iterable[str]) -> set[str]:
        covered_set = set(covered)
        needed = set(self.answer_nodes)
        nodes = self.node_map
        for edge in self.required_edges:
            if edge.id in covered_set:
                continue
            if nodes[edge.source].is_variable:
                needed.add(edge.source)
            if nodes[edge.target].is_variable:
                needed.add(edge.target)
        return needed


@dataclass(frozen=True)
class Realization:
    edge_id: str
    fact_ids: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]
    provenance: tuple[str, ...]
    relation_score: float
    endpoint_score: float
    ontology_score: float
    score: float

    @property
    def binding_map(self) -> dict[str, str]:
        return dict(self.bindings)

    @property
    def length(self) -> int:
        return len(self.fact_ids)

    @property
    def signature(self) -> tuple[Any, ...]:
        return self.edge_id, self.fact_ids, self.bindings


@dataclass(frozen=True)
class MatchState:
    covered: frozenset[str]
    bindings: tuple[tuple[str, str], ...]
    realizations: tuple[Realization, ...]
    score: float

    @property
    def binding_map(self) -> dict[str, str]:
        return dict(self.bindings)

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            tuple(sorted(self.covered)),
            self.bindings,
            tuple(item.signature for item in self.realizations),
        )


@dataclass(frozen=True)
class Certificate:
    complete: bool
    coverage: bool
    binding: bool
    projection: bool
    provenance: bool
    reasons: tuple[str, ...] = ()


@dataclass
class ExecutionTrace:
    missing_edges: list[str] = field(default_factory=list)
    failed_certificates: dict[str, list[str]] = field(default_factory=dict)
    feasible_bindings: dict[str, list[str]] = field(default_factory=dict)
    best_partial_score: float = 0.0


@dataclass(frozen=True)
class CertifiedMatch:
    state: MatchState
    certificate: Certificate


def labels_from_mapping(values: Mapping[str, float] | None) -> tuple[OntologyLabel, ...]:
    if not values:
        return ()
    rows = [
        OntologyLabel(str(name), max(0.0, min(1.0, float(score))))
        for name, score in values.items()
        if str(name).strip()
    ]
    return tuple(sorted(rows, key=lambda item: (-item.confidence, item.name)))

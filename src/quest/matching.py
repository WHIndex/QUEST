from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .config import SearchConfig
from .evidence import EvidenceGraph
from .model import EvidenceFact, QueryEdge, QueryGraph, QueryNode, Realization
from .ontology import OntologyRegistry
from .semantic import SemanticScorer


@dataclass
class SearchResult:
    cached: dict[str, list[Realization]]
    active: dict[str, list[Realization]]
    domains: dict[str, set[str]]
    edge_order: list[str]
    observed_cardinality: dict[str, float]
    operations: list[dict[str, object]] = field(default_factory=list)


class RealizationGenerator:
    def __init__(
        self,
        graph: EvidenceGraph,
        ontology: OntologyRegistry,
        scorer: SemanticScorer,
        config: SearchConfig,
    ):
        self.graph = graph
        self.ontology = ontology
        self.scorer = scorer
        self.config = config

    def direct(
        self,
        query: QueryGraph,
        edge: QueryEdge,
        domains: dict[str, set[str]],
        limit: int | None = None,
    ) -> list[Realization]:
        limit = limit or self.config.direct_budget
        source = query.node_map[edge.source]
        target = query.node_map[edge.target]
        source_values = self._endpoint_values(source, domains)
        target_values = self._endpoint_values(target, domains)

        candidates: dict[str, EvidenceFact] = {}
        if source_values:
            for entity in source_values:
                for fact in self.graph.adjacent(entity, direction="out"):
                    candidates[fact.id] = fact
        elif target_values:
            for entity in target_values:
                for fact in self.graph.adjacent(entity, direction="in"):
                    candidates[fact.id] = fact
        else:
            for fact, _ in self.graph.rank_facts(edge.relation, limit * 4):
                candidates[fact.id] = fact

        rows = []
        for fact in candidates.values():
            realization = self._direct_realization(
                query, edge, fact, source_values, target_values
            )
            if realization is not None:
                rows.append(realization)
        return _top_unique(rows, limit)

    def paths(
        self,
        query: QueryGraph,
        edge: QueryEdge,
        domains: dict[str, set[str]],
        limit: int | None = None,
    ) -> list[Realization]:
        limit = limit or self.config.path_budget
        source = query.node_map[edge.source]
        target = query.node_map[edge.target]
        source_values = self._endpoint_values(source, domains)
        target_values = self._endpoint_values(target, domains)
        if not source_values:
            seed_facts = self.graph.rank_facts(edge.relation, min(limit, 24))
            source_values = {self.graph.entity_key(fact.subject) for fact, _ in seed_facts}

        completed: list[Realization] = []
        frontier: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
            (entity, (), (entity,)) for entity in source_values
        ]
        for depth in range(1, self.config.max_path_length + 1):
            next_frontier: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
            for entity, fact_ids, visited in frontier:
                outgoing = self.graph.adjacent(entity, direction="out")
                outgoing.sort(
                    key=lambda fact: self.scorer.similarity(edge.relation, fact.relation),
                    reverse=True,
                )
                for fact in outgoing[: self.config.path_budget]:
                    destination = self.graph.entity_key(fact.object)
                    if destination in visited:
                        continue
                    path_ids = fact_ids + (fact.id,)
                    relation_chain = self._relation_chain(path_ids)
                    prefix_score = self.scorer.similarity(edge.relation, relation_chain)
                    if prefix_score < self.config.path_prefix_threshold:
                        continue
                    if depth >= 2:
                        realization = self._path_realization(
                            query,
                            edge,
                            path_ids,
                            source_values,
                            target_values,
                        )
                        if realization is not None:
                            completed.append(realization)
                    if depth < self.config.max_path_length:
                        next_frontier.append(
                            (destination, path_ids, visited + (destination,))
                        )
            next_frontier.sort(
                key=lambda row: self.scorer.similarity(
                    edge.relation, self._relation_chain(row[1])
                ),
                reverse=True,
            )
            frontier = next_frontier[: self.config.path_budget]
            if not frontier and depth < self.config.max_path_length:
                break
        return _top_unique(completed, limit)

    def realizations_from_facts(
        self,
        query: QueryGraph,
        edge: QueryEdge,
        facts: Iterable[EvidenceFact],
        domains: dict[str, set[str]],
    ) -> list[Realization]:
        source_values = self._endpoint_values(query.node_map[edge.source], domains)
        target_values = self._endpoint_values(query.node_map[edge.target], domains)
        rows = []
        for fact in facts:
            item = self._direct_realization(
                query, edge, fact, source_values, target_values
            )
            if item is not None:
                rows.append(item)
        return _top_unique(rows, self.config.path_budget)

    def _direct_realization(
        self,
        query: QueryGraph,
        edge: QueryEdge,
        fact: EvidenceFact,
        source_values: set[str],
        target_values: set[str],
    ) -> Realization | None:
        relation_score = self.scorer.similarity(edge.relation, fact.relation)
        if relation_score < self.config.relation_threshold:
            return None
        source_key = self.graph.entity_key(fact.subject)
        target_key = self.graph.entity_key(fact.object)
        if source_values and source_key not in source_values:
            return None
        if target_values and target_key not in target_values:
            return None
        return self._make_realization(
            query,
            edge,
            (fact.id,),
            source_key,
            target_key,
            relation_score,
            fact.subject_types,
            fact.object_types,
        )

    def _path_realization(
        self,
        query: QueryGraph,
        edge: QueryEdge,
        fact_ids: tuple[str, ...],
        source_values: set[str],
        target_values: set[str],
    ) -> Realization | None:
        facts = [self.graph.facts[fact_id] for fact_id in fact_ids]
        for left, right in zip(facts, facts[1:]):
            if self.graph.entity_key(left.object) != self.graph.entity_key(right.subject):
                return None
        source_key = self.graph.entity_key(facts[0].subject)
        target_key = self.graph.entity_key(facts[-1].object)
        if source_values and source_key not in source_values:
            return None
        if target_values and target_key not in target_values:
            return None
        chain = self._relation_chain(fact_ids)
        relation_score = self.scorer.similarity(edge.relation, chain)
        if relation_score < self.config.relation_threshold:
            return None
        return self._make_realization(
            query,
            edge,
            fact_ids,
            source_key,
            target_key,
            relation_score,
            facts[0].subject_types,
            facts[-1].object_types,
        )

    def _make_realization(
        self,
        query: QueryGraph,
        edge: QueryEdge,
        fact_ids: tuple[str, ...],
        source_key: str,
        target_key: str,
        relation_score: float,
        source_types,
        target_types,
    ) -> Realization | None:
        source = query.node_map[edge.source]
        target = query.node_map[edge.target]
        endpoint_scores = []
        if not source.is_variable:
            endpoint_scores.append(
                self.scorer.similarity(source.label, self.graph.entity_label(source_key))
            )
        if not target.is_variable:
            endpoint_scores.append(
                self.scorer.similarity(target.label, self.graph.entity_label(target_key))
            )
        endpoint_score = sum(endpoint_scores) / len(endpoint_scores) if endpoint_scores else 1.0
        if endpoint_score < self.config.endpoint_threshold:
            return None

        ontology_scores = [
            self.ontology.compatibility(source.ontology, source_types),
            self.ontology.compatibility(target.ontology, target_types),
        ]
        ontology_score = sum(ontology_scores) / len(ontology_scores)
        if ontology_score < self.config.ontology_threshold and endpoint_score < 0.72:
            return None

        bindings = []
        if source.is_variable:
            bindings.append((source.id, source_key))
        if target.is_variable:
            bindings.append((target.id, target_key))
        provenance = tuple(
            dict.fromkeys(self.graph.facts[fact_id].passage_id for fact_id in fact_ids)
        )
        weights = self.config.weights
        score = (
            weights.relation * relation_score
            + weights.endpoint * endpoint_score
            + weights.ontology * ontology_score
            - weights.length * (len(fact_ids) - 1)
        )
        return Realization(
            edge_id=edge.id,
            fact_ids=fact_ids,
            bindings=tuple(sorted(bindings)),
            provenance=provenance,
            relation_score=relation_score,
            endpoint_score=endpoint_score,
            ontology_score=ontology_score,
            score=score,
        )

    def _endpoint_values(
        self, node: QueryNode, domains: dict[str, set[str]]
    ) -> set[str]:
        if node.is_variable:
            return set(domains.get(node.id, ()))
        resolved = self.graph.resolve_entity(node.grounded_id or node.label)
        return {resolved} if resolved else set()

    def _relation_chain(self, fact_ids: tuple[str, ...]) -> str:
        return " then ".join(self.graph.facts[fact_id].relation for fact_id in fact_ids)


class CostGuidedBindingSearch:
    def __init__(
        self,
        graph: EvidenceGraph,
        generator: RealizationGenerator,
        config: SearchConfig,
    ):
        self.graph = graph
        self.generator = generator
        self.config = config

    def run(self, query: QueryGraph) -> SearchResult:
        cached = {edge.id: [] for edge in query.required_edges}
        active = {edge.id: [] for edge in query.required_edges}
        domains: dict[str, set[str]] = {}
        unresolved = {edge.id for edge in query.required_edges}
        observed: dict[str, float] = {}
        edge_order: list[str] = []
        operations: list[dict[str, object]] = []

        while unresolved:
            edge = min(
                (query.edge_map[edge_id] for edge_id in unresolved),
                key=lambda item: (self.estimate_cost(query, item, domains, observed), item.id),
            )
            rows = self.generator.direct(query, edge, domains)
            cached[edge.id] = _merge_realizations(cached[edge.id], rows)
            active, domains = refresh_active_sets(query, cached)
            observed[edge.id] = self._update_observed(
                observed.get(edge.id), len(rows)
            )
            unresolved.remove(edge.id)
            edge_order.append(edge.id)
            operations.append(
                {
                    "operation": "direct",
                    "edge": edge.id,
                    "examined": len(rows),
                    "cost": self.estimate_cost(query, edge, domains, observed),
                }
            )

        expanded: set[str] = set()
        while not candidate_sufficient(query, active):
            targets = [
                edge
                for edge in query.required_edges
                if edge.id not in expanded
                and (
                    not active[edge.id]
                    or max((row.relation_score for row in active[edge.id]), default=0.0) < 0.68
                )
            ]
            if not targets:
                targets = [
                    edge for edge in query.required_edges if edge.id not in expanded
                ]
            if not targets:
                break
            edge = min(
                targets,
                key=lambda item: (self.estimate_cost(query, item, domains, observed), item.id),
            )
            paths = self.generator.paths(query, edge, domains)
            cached[edge.id] = _merge_realizations(cached[edge.id], paths)
            active, domains = refresh_active_sets(query, cached)
            expanded.add(edge.id)
            operations.append(
                {
                    "operation": "path",
                    "edge": edge.id,
                    "examined": len(paths),
                    "max_length": self.config.max_path_length,
                }
            )

        return SearchResult(
            cached=cached,
            active=active,
            domains=domains,
            edge_order=edge_order,
            observed_cardinality=observed,
            operations=operations,
        )

    def estimate_cost(
        self,
        query: QueryGraph,
        edge: QueryEdge,
        domains: dict[str, set[str]],
        observed: dict[str, float],
    ) -> float:
        node_map = query.node_map
        estimates = []
        for node_id, direction in ((edge.source, "out"), (edge.target, "in")):
            node = node_map[node_id]
            values = domains.get(node_id, set()) if node.is_variable else set()
            if not node.is_variable:
                resolved = self.graph.resolve_entity(node.grounded_id or node.label)
                values = {resolved} if resolved else set()
            if values:
                counts = [
                    len(self.graph.outgoing[value])
                    if direction == "out"
                    else len(self.graph.incoming[value])
                    for value in values
                ]
                estimates.append(sum(counts) / max(1, len(counts)))
        initial = min(estimates) if estimates else float(
            max(1, self.graph.direct_cardinality(edge.relation))
        )
        if edge.id not in observed:
            return initial
        alpha = self.config.runtime_cost_alpha
        return (1.0 - alpha) * initial + alpha * observed[edge.id]

    def _update_observed(self, previous: float | None, value: int) -> float:
        if previous is None:
            return float(value)
        alpha = self.config.runtime_cost_alpha
        return (1.0 - alpha) * previous + alpha * value


def candidate_sufficient(
    query: QueryGraph, active: dict[str, list[Realization]]
) -> bool:
    if any(not active.get(edge.id) for edge in query.required_edges):
        return False
    for variable in _shared_variables(query):
        domains = []
        for edge in query.required_edges:
            if variable not in {edge.source, edge.target}:
                continue
            values = {
                row.binding_map[variable]
                for row in active.get(edge.id, [])
                if variable in row.binding_map
            }
            if values:
                domains.append(values)
        if domains and not set.intersection(*domains):
            return False
    return True


def refresh_active_sets(
    query: QueryGraph,
    cached: dict[str, list[Realization]],
) -> tuple[dict[str, list[Realization]], dict[str, set[str]]]:
    active = {edge_id: list(rows) for edge_id, rows in cached.items()}
    shared = _shared_variables(query)
    changed = True
    while changed:
        changed = False
        for variable in shared:
            edge_ids = [
                edge.id
                for edge in query.required_edges
                if variable in {edge.source, edge.target} and active.get(edge.id)
            ]
            value_sets = [
                {
                    row.binding_map[variable]
                    for row in active[edge_id]
                    if variable in row.binding_map
                }
                for edge_id in edge_ids
            ]
            value_sets = [values for values in value_sets if values]
            if len(value_sets) < 2:
                continue
            feasible = set.intersection(*value_sets)
            for edge_id in edge_ids:
                filtered = [
                    row
                    for row in active[edge_id]
                    if variable not in row.binding_map
                    or row.binding_map[variable] in feasible
                ]
                if len(filtered) != len(active[edge_id]):
                    active[edge_id] = filtered
                    changed = True

    domains: dict[str, set[str]] = defaultdict(set)
    for rows in active.values():
        for row in rows:
            for variable, entity in row.bindings:
                domains[variable].add(entity)
    return active, dict(domains)


def _shared_variables(query: QueryGraph) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    nodes = query.node_map
    for edge in query.required_edges:
        for node_id in (edge.source, edge.target):
            if nodes[node_id].is_variable:
                counts[node_id] += 1
    return {node_id for node_id, count in counts.items() if count > 1}


def _merge_realizations(
    current: list[Realization], additions: Iterable[Realization]
) -> list[Realization]:
    best = {row.signature: row for row in current}
    for row in additions:
        existing = best.get(row.signature)
        if existing is None or row.score > existing.score:
            best[row.signature] = row
    return sorted(best.values(), key=lambda row: (-row.score, row.signature))


def _top_unique(rows: Iterable[Realization], limit: int) -> list[Realization]:
    return _merge_realizations([], rows)[:limit]

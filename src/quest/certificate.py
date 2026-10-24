from __future__ import annotations

import re
from collections.abc import Sequence

from .evidence import EvidenceGraph
from .model import Certificate, CertifiedMatch, MatchState, QueryGraph
from .semantic import SemanticScorer, normalize_text


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


class CertificateChecker:
    def __init__(
        self,
        graph: EvidenceGraph,
        scorer: SemanticScorer,
        support_threshold: float = 0.34,
    ):
        self.graph = graph
        self.scorer = scorer
        self.support_threshold = support_threshold

    def certify_all(
        self,
        query: QueryGraph,
        terminal: Sequence[MatchState],
        limit: int,
    ) -> list[CertifiedMatch]:
        rows = []
        for state in terminal:
            certificate = self.certify(query, state, terminal)
            if certificate.complete:
                rows.append(CertifiedMatch(state, certificate))
        rows.sort(key=lambda row: (-row.state.score, row.state.signature))
        return rows[:limit]

    def certify(
        self,
        query: QueryGraph,
        state: MatchState,
        terminal: Sequence[MatchState],
    ) -> Certificate:
        required = {edge.id for edge in query.required_edges}
        coverage = required.issubset(state.covered) and self._operators_hold(
            query, state, terminal
        )
        binding = self._bindings_hold(query, state)
        projection = self._projection_holds(query, state)
        provenance = self._provenance_holds(query, state)
        reasons = []
        if not coverage:
            reasons.append("constraint_coverage_failed")
        if not binding:
            reasons.append("inconsistent_variable_binding")
        if not projection:
            reasons.append("unbound_answer_projection")
        if not provenance:
            reasons.append("unsupported_provenance")
        return Certificate(
            complete=coverage and binding and projection and provenance,
            coverage=coverage,
            binding=binding,
            projection=projection,
            provenance=provenance,
            reasons=tuple(reasons),
        )

    def _bindings_hold(self, query: QueryGraph, state: MatchState) -> bool:
        seen: dict[str, str] = {}
        edge_map = query.edge_map
        for realization in state.realizations:
            if realization.edge_id not in edge_map:
                return False
            for variable, entity in realization.bindings:
                if variable in seen and seen[variable] != entity:
                    return False
                seen[variable] = entity
        return True

    def _projection_holds(self, query: QueryGraph, state: MatchState) -> bool:
        bindings = state.binding_map
        nodes = query.node_map
        for node_id in query.answer_nodes:
            node = nodes[node_id]
            if node.is_variable and node_id not in bindings:
                return False
        return True

    def _provenance_holds(self, query: QueryGraph, state: MatchState) -> bool:
        edge_map = query.edge_map
        for realization in state.realizations:
            edge = edge_map[realization.edge_id]
            facts = []
            for fact_id in realization.fact_ids:
                fact = self.graph.facts.get(fact_id)
                if fact is None or fact.passage_id not in self.graph.passages:
                    return False
                if fact.sentence not in self.graph.passages[fact.passage_id].text:
                    return False
                statement = f"{fact.subject} {fact.relation} {fact.object}"
                support = self.scorer.similarity(statement, fact.sentence)
                if support < self.support_threshold:
                    return False
                if normalize_text(fact.subject) == normalize_text(fact.object):
                    return False
                facts.append(fact)
            for left, right in zip(facts, facts[1:]):
                if self.graph.entity_key(left.object) != self.graph.entity_key(right.subject):
                    return False
            relation_chain = " then ".join(fact.relation for fact in facts)
            if self.scorer.similarity(edge.relation, relation_chain) < 0.34:
                return False
        return True

    def _operators_hold(
        self,
        query: QueryGraph,
        state: MatchState,
        terminal: Sequence[MatchState],
    ) -> bool:
        if not query.operators:
            return True
        for operator in query.operators:
            try:
                if not self._operator_holds(query, state, terminal, operator):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _operator_holds(self, query, state, terminal, operator) -> bool:
        name = normalize_text(operator.name).replace(" ", "_")
        values = [self._argument(query, state, item) for item in operator.arguments]
        if any(value is None for value in values):
            return False
        if name in {"equal", "equals", "eq", "same_as"}:
            return len(values) >= 2 and normalize_text(values[0]) == normalize_text(values[1])
        if name in {"not_equal", "neq", "different_from"}:
            return len(values) >= 2 and normalize_text(values[0]) != normalize_text(values[1])
        if name in {"greater_than", "gt", "after"}:
            return len(values) >= 2 and self._numeric(values[0]) > self._numeric(values[1])
        if name in {"less_than", "lt", "before"}:
            return len(values) >= 2 and self._numeric(values[0]) < self._numeric(values[1])
        if name == "count":
            variable = operator.arguments[0] if operator.arguments else ""
            distinct = {
                item.binding_map.get(variable)
                for item in terminal
                if item.binding_map.get(variable) is not None
            }
            return operator.value is None or len(distinct) == int(operator.value)
        if name in {"min", "argmin", "max", "argmax"}:
            variable = operator.arguments[0] if operator.arguments else ""
            current = self._numeric(state.binding_map.get(variable, ""))
            candidates = [
                self._numeric(item.binding_map[variable])
                for item in terminal
                if variable in item.binding_map
            ]
            if not candidates:
                return False
            return current == (min(candidates) if name in {"min", "argmin"} else max(candidates))
        return False

    @staticmethod
    def _argument(query: QueryGraph, state: MatchState, name: str) -> str | None:
        if name in state.binding_map:
            return state.binding_map[name]
        node = query.node_map.get(name)
        if node is not None and not node.is_variable:
            return node.grounded_id or node.label
        return name or None

    @staticmethod
    def _numeric(value: str) -> float:
        match = _NUMBER.search(value.replace(",", ""))
        if not match:
            raise ValueError(f"operator argument is not numeric: {value}")
        return float(match.group())

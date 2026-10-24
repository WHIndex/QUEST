from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .certificate import CertificateChecker
from .client import ModelClient
from .config import SearchConfig
from .dp import BindDP, DPResult, changed_active_edges
from .evidence import CorpusBuilder, EvidenceGraph
from .matching import RealizationGenerator, SearchResult, refresh_active_sets
from .model import CertifiedMatch, EvidenceFact, ExecutionTrace, QueryEdge, QueryGraph
from .ontology import OntologyRegistry
from .semantic import SemanticScorer, normalize_text


@dataclass(frozen=True)
class RetrievalRequest:
    edge_id: str
    query: str
    bindings: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RecoveryResult:
    search: SearchResult
    dp: DPResult
    certified: tuple[CertifiedMatch, ...]
    rounds: int
    requests: int
    added_realizations: int
    reused_subsets: int
    recomputed_subsets: int


class RetrievalAgent:
    def __init__(self, client: ModelClient):
        self.client = client

    def plan(
        self,
        question: str,
        query: QueryGraph,
        trace: ExecutionTrace,
    ) -> list[RetrievalRequest]:
        edge_rows = [
            {
                "id": edge.id,
                "source": query.node_map[edge.source].label,
                "relation": edge.relation,
                "target": query.node_map[edge.target].label,
            }
            for edge in query.required_edges
            if edge.id in trace.missing_edges
        ]
        response = self.client.complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Create targeted source-retrieval requests for the unmet query constraints. "
                        "Use the available variable bindings to make each request specific. Return a "
                        "requests array with edge_id, query, and bindings. Each query must seek textual "
                        "evidence for exactly one listed directed constraint. Do not answer the question."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "unmet_constraints": edge_rows,
                            "feasible_bindings": trace.feasible_bindings,
                            "certificate_failures": trace.failed_certificates,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        allowed = {row["id"] for row in edge_rows}
        requests = []
        for row in response.get("requests", []):
            if not isinstance(row, dict):
                continue
            edge_id = str(row.get("edge_id", ""))
            text = str(row.get("query", "")).strip()
            if edge_id not in allowed or not text:
                continue
            bindings_value = row.get("bindings", {})
            if isinstance(bindings_value, dict):
                bindings = tuple(
                    sorted((str(key), str(value)) for key, value in bindings_value.items())
                )
            else:
                bindings = ()
            requests.append(RetrievalRequest(edge_id, text, bindings))
        return requests


class EvidenceCompletion:
    def __init__(
        self,
        graph: EvidenceGraph,
        ontology: OntologyRegistry,
        scorer: SemanticScorer,
        builder: CorpusBuilder,
        agent: RetrievalAgent,
        generator: RealizationGenerator,
        dp: BindDP,
        checker: CertificateChecker,
        config: SearchConfig,
    ):
        self.graph = graph
        self.ontology = ontology
        self.scorer = scorer
        self.builder = builder
        self.agent = agent
        self.generator = generator
        self.dp = dp
        self.checker = checker
        self.config = config

    def recover(
        self,
        question: str,
        query: QueryGraph,
        search: SearchResult,
        dp_result: DPResult,
    ) -> RecoveryResult:
        certified = self.checker.certify_all(
            query, dp_result.terminal, self.config.match_budget
        )
        total_requests = 0
        total_added = 0
        reused = 0
        recomputed = 0
        rounds = 0

        while not certified and rounds < self.config.recovery_rounds:
            rounds += 1
            trace = build_execution_trace(query, dp_result, self.checker)
            requests = self.agent.plan(question, query, trace)
            if not requests:
                break
            total_requests += len(requests)
            before_active = {
                edge_id: list(rows) for edge_id, rows in search.active.items()
            }
            added_this_round = 0

            for request in requests:
                edge = query.edge_map[request.edge_id]
                new_facts = self._retrieve_and_verify(
                    query, edge, request, dict(request.bindings)
                )
                if not new_facts:
                    continue
                self.graph.add_facts(new_facts)
                recovery_domains = {
                    variable: set(values)
                    for variable, values in trace.feasible_bindings.items()
                }
                direct = self.generator.realizations_from_facts(
                    query, edge, new_facts, recovery_domains
                )
                paths = self.generator.paths(
                    query, edge, recovery_domains, self.config.path_budget
                )
                additions = direct + paths
                if additions:
                    search.cached[edge.id] = _merge(
                        search.cached.get(edge.id, []), additions
                    )
                    added_this_round += len(additions)

            if added_this_round == 0:
                break
            search.active, search.domains = refresh_active_sets(query, search.cached)
            changed = changed_active_edges(before_active, search.active)
            if not changed:
                break
            dp_result = self.dp.delta_update(
                query, search.active, dp_result, changed
            )
            certified = self.checker.certify_all(
                query, dp_result.terminal, self.config.match_budget
            )
            total_added += added_this_round
            reused += dp_result.reused_subsets
            recomputed += dp_result.recomputed_subsets
        return RecoveryResult(
            search=search,
            dp=dp_result,
            certified=tuple(certified),
            rounds=rounds,
            requests=total_requests,
            added_realizations=total_added,
            reused_subsets=reused,
            recomputed_subsets=recomputed,
        )

    def _retrieve_and_verify(
        self,
        query: QueryGraph,
        edge: QueryEdge,
        request: RetrievalRequest,
        bindings: dict[str, str],
    ) -> list[EvidenceFact]:
        passages = self.graph.rank_passages(
            request.query, self.config.recovery_passages
        )
        candidates = []
        seen = set(self.graph.facts)
        for passage, _ in passages:
            for fact in self.builder.extract_for_constraint(
                passage,
                source=query.node_map[edge.source].label,
                relation=edge.relation,
                target=query.node_map[edge.target].label,
                retrieval_query=request.query,
                bindings=bindings,
            ):
                if fact.id in seen:
                    continue
                annotated = self.ontology.annotate_fact(fact)
                if self._valid_fact(edge, annotated, bindings):
                    candidates.append(annotated)
                    seen.add(annotated.id)
        return candidates

    def _valid_fact(
        self,
        edge: QueryEdge,
        fact: EvidenceFact,
        bindings: dict[str, str],
    ) -> bool:
        if fact.passage_id not in self.graph.passages:
            return False
        passage = self.graph.passages[fact.passage_id]
        if fact.sentence not in passage.text:
            return False
        if self.scorer.similarity(edge.relation, fact.relation) < self.config.relation_threshold:
            return False
        relation_ontology = self.ontology.compatibility(
            edge.ontology, fact.relation_types
        )
        if relation_ontology < self.config.ontology_threshold:
            return False
        source_bound = bindings.get(edge.source)
        target_bound = bindings.get(edge.target)
        if source_bound and normalize_text(source_bound) != normalize_text(fact.subject):
            return False
        if target_bound and normalize_text(target_bound) != normalize_text(fact.object):
            return False
        statement = f"{fact.subject} {fact.relation} {fact.object}"
        return self.scorer.similarity(statement, fact.sentence) >= 0.34


def build_execution_trace(
    query: QueryGraph,
    dp_result: DPResult,
    checker: CertificateChecker,
) -> ExecutionTrace:
    states = [
        state
        for groups in dp_result.table.values()
        for rows in groups.values()
        for state in rows
    ]
    best = max(states, key=lambda row: (len(row.covered), row.score), default=None)
    required = {edge.id for edge in query.required_edges}
    missing = sorted(required - (set(best.covered) if best else set()))
    failures: dict[str, list[str]] = {}
    if dp_result.terminal:
        for index, state in enumerate(dp_result.terminal):
            certificate = checker.certify(query, state, dp_result.terminal)
            if not certificate.complete:
                failures[str(index)] = list(certificate.reasons)
        if failures and not missing:
            missing = sorted(required)

    bindings: dict[str, set[str]] = {}
    for state in states:
        for variable, entity in state.bindings:
            bindings.setdefault(variable, set()).add(entity)
    return ExecutionTrace(
        missing_edges=missing,
        failed_certificates=failures,
        feasible_bindings={key: sorted(values) for key, values in bindings.items()},
        best_partial_score=best.score if best else 0.0,
    )


def _merge(current, additions):
    rows = {item.signature: item for item in current}
    for item in additions:
        old = rows.get(item.signature)
        if old is None or item.score > old.score:
            rows[item.signature] = item
    return sorted(rows.values(), key=lambda item: (-item.score, item.signature))

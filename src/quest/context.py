from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .evidence import EvidenceGraph
from .model import CertifiedMatch, Passage, QueryGraph


@dataclass(frozen=True)
class ContextSelection:
    passages: tuple[Passage, ...]
    covered_edges: frozenset[str]
    characters: int


class ContextSelector:
    def __init__(self, graph: EvidenceGraph, passage_budget: int, character_budget: int):
        self.graph = graph
        self.passage_budget = passage_budget
        self.character_budget = character_budget

    def select(
        self,
        query: QueryGraph,
        matches: Sequence[CertifiedMatch],
    ) -> ContextSelection:
        selected: list[Passage] = []
        selected_ids: set[str] = set()
        covered: set[str] = set()
        characters = 0

        for match in sorted(matches, key=lambda row: -row.state.score):
            candidates: dict[str, set[str]] = {}
            for realization in match.state.realizations:
                for passage_id in realization.provenance:
                    candidates.setdefault(passage_id, set()).add(realization.edge_id)

            while candidates and len(selected) < self.passage_budget:
                ranked = []
                for passage_id, edges in candidates.items():
                    if passage_id in selected_ids or passage_id not in self.graph.passages:
                        continue
                    passage = self.graph.passages[passage_id]
                    new_edges = edges - covered
                    answer_bonus = sum(
                        1
                        for edge_id in new_edges
                        if any(
                            node_id in query.answer_nodes
                            for node_id in (
                                query.edge_map[edge_id].source,
                                query.edge_map[edge_id].target,
                            )
                        )
                    )
                    utility = 2.0 * len(new_edges) + answer_bonus
                    if utility <= 0.0:
                        continue
                    ranked.append((utility / max(1, len(passage.text)), passage_id, edges))
                if not ranked:
                    break
                _, passage_id, edges = max(ranked)
                passage = self.graph.passages[passage_id]
                if characters + len(passage.text) > self.character_budget:
                    candidates.pop(passage_id, None)
                    continue
                selected.append(passage)
                selected_ids.add(passage_id)
                covered.update(edges)
                characters += len(passage.text)
                candidates.pop(passage_id, None)

            if len(selected) >= self.passage_budget or characters >= self.character_budget:
                break
        return ContextSelection(tuple(selected), frozenset(covered), characters)

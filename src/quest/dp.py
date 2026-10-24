from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from .config import SearchConfig
from .model import MatchState, QueryGraph, Realization


DPTable = dict[frozenset[str], dict[tuple[tuple[str, str], ...], list[MatchState]]]


@dataclass(frozen=True)
class DPResult:
    terminal: tuple[MatchState, ...]
    table: DPTable
    reused_subsets: int = 0
    recomputed_subsets: int = 0


class BindDP:
    def __init__(self, config: SearchConfig):
        self.config = config

    def run(
        self,
        query: QueryGraph,
        active: dict[str, list[Realization]],
    ) -> DPResult:
        table: DPTable = {
            frozenset(): {
                (): [MatchState(frozenset(), (), (), 0.0)]
            }
        }
        edge_ids = tuple(edge.id for edge in query.required_edges)
        for size in range(1, len(edge_ids) + 1):
            for values in combinations(edge_ids, size):
                subset = frozenset(values)
                table[subset] = self._compute_subset(query, active, table, subset)
        terminal = self._terminal(table, frozenset(edge_ids))
        return DPResult(tuple(terminal), table, recomputed_subsets=max(0, len(table) - 1))

    def delta_update(
        self,
        query: QueryGraph,
        active: dict[str, list[Realization]],
        previous: DPResult,
        changed_edges: set[str],
    ) -> DPResult:
        edge_ids = tuple(edge.id for edge in query.required_edges)
        table: DPTable = {}
        reused = 0
        recomputed = 0
        for size in range(0, len(edge_ids) + 1):
            for values in combinations(edge_ids, size):
                subset = frozenset(values)
                affected = bool(subset & changed_edges)
                if not affected and subset in previous.table:
                    table[subset] = previous.table[subset]
                    reused += 1
                    continue
                if not subset:
                    table[subset] = {
                        (): [MatchState(frozenset(), (), (), 0.0)]
                    }
                else:
                    table[subset] = self._compute_subset(query, active, table, subset)
                recomputed += 1
        terminal = self._terminal(table, frozenset(edge_ids))
        return DPResult(tuple(terminal), table, reused, recomputed)

    def _compute_subset(
        self,
        query: QueryGraph,
        active: dict[str, list[Realization]],
        table: DPTable,
        subset: frozenset[str],
    ) -> dict[tuple[tuple[str, str], ...], list[MatchState]]:
        grouped: dict[tuple[tuple[str, str], ...], list[MatchState]] = defaultdict(list)
        for edge_id in sorted(subset):
            predecessor = subset - {edge_id}
            for states in table.get(predecessor, {}).values():
                for state in states:
                    for realization in active.get(edge_id, []):
                        extended = self._extend(query, state, realization)
                        if extended is not None:
                            grouped[extended.bindings].append(extended)

        result = {}
        for binding_key, states in grouped.items():
            unique: dict[tuple[object, ...], MatchState] = {}
            for state in states:
                existing = unique.get(state.signature)
                if existing is None or state.score > existing.score:
                    unique[state.signature] = state
            result[binding_key] = sorted(
                unique.values(), key=lambda row: (-row.score, row.signature)
            )[: self.config.state_budget]
        return result

    def _extend(
        self,
        query: QueryGraph,
        state: MatchState,
        realization: Realization,
    ) -> MatchState | None:
        if realization.edge_id in state.covered:
            return None
        bindings = state.binding_map
        for variable, entity in realization.bindings:
            if variable in bindings and bindings[variable] != entity:
                return None
            bindings[variable] = entity
        covered = state.covered | {realization.edge_id}
        needed = query.variables_needed_after(covered)
        projected = tuple(sorted((key, value) for key, value in bindings.items() if key in needed))
        return MatchState(
            covered=covered,
            bindings=projected,
            realizations=tuple(
                sorted(state.realizations + (realization,), key=lambda row: row.edge_id)
            ),
            score=state.score + realization.score,
        )

    def _terminal(
        self, table: DPTable, complete: frozenset[str]
    ) -> list[MatchState]:
        rows = [state for states in table.get(complete, {}).values() for state in states]
        rows.sort(key=lambda row: (-row.score, row.signature))
        return rows[: self.config.match_budget]


def changed_active_edges(
    before: dict[str, list[Realization]],
    after: dict[str, list[Realization]],
) -> set[str]:
    changed = set()
    for edge_id in set(before) | set(after):
        left = {row.signature for row in before.get(edge_id, [])}
        right = {row.signature for row in after.get(edge_id, [])}
        if left != right:
            changed.add(edge_id)
    return changed

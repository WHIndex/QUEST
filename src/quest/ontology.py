from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import replace
import json
from pathlib import Path

from .model import EvidenceFact, OntologyLabel, QueryEdge, QueryNode
from .semantic import SemanticScorer, normalize_text


DEFAULT_ENTITY_CONCEPTS = {
    "entity": (),
    "person": ("person", "author", "actor", "director", "founder", "scientist"),
    "organization": ("organization", "company", "school", "team", "agency"),
    "location": ("location", "country", "city", "state", "region", "place"),
    "work": ("work", "book", "film", "song", "album", "publication"),
    "event": ("event", "war", "election", "tournament", "ceremony"),
    "time": ("time", "date", "year", "period"),
    "quantity": ("quantity", "number", "amount", "score", "population"),
}

DEFAULT_RELATION_CONCEPTS = {
    "relation": (),
    "authorship": ("wrote", "written by", "author of", "created by"),
    "location": ("located in", "country", "city", "place", "headquartered in"),
    "birth": ("born in", "birthplace", "date of birth"),
    "affiliation": ("worked at", "member of", "employed by", "founded"),
    "part_whole": ("part of", "contains", "belongs to"),
    "family": ("parent", "child", "spouse", "sibling", "grandparent"),
    "time": ("before", "after", "during", "founded in", "released in"),
    "comparison": ("greater than", "less than", "same as", "different from"),
}


class OntologyRegistry:
    def __init__(
        self,
        scorer: SemanticScorer,
        entity_concepts: dict[str, tuple[str, ...]] | None = None,
        relation_concepts: dict[str, tuple[str, ...]] | None = None,
    ):
        self.scorer = scorer
        self.entity_concepts = dict(entity_concepts or DEFAULT_ENTITY_CONCEPTS)
        self.relation_concepts = dict(relation_concepts or DEFAULT_RELATION_CONCEPTS)
        self.entity_refinements: dict[str, str] = {}
        self.relation_refinements: dict[str, str] = {}

    def induce(self, facts: Iterable[EvidenceFact], min_support: int = 3) -> None:
        entity_context: dict[str, Counter[str]] = defaultdict(Counter)
        relation_counts: Counter[str] = Counter()
        for fact in facts:
            relation = normalize_text(fact.relation)
            relation_counts[relation] += 1
            entity_context[normalize_text(fact.subject)][relation] += 1
            entity_context[normalize_text(fact.object)][relation] += 1

        for entity, context in entity_context.items():
            if sum(context.values()) < min_support:
                continue
            description = entity + " " + " ".join(
                relation for relation, _ in context.most_common(4)
            )
            label = self._best_concept(description, self.entity_concepts)
            if label != "entity":
                self.entity_refinements[entity] = label

        for relation, count in relation_counts.items():
            if count < min_support:
                continue
            label = self._best_concept(relation, self.relation_concepts)
            if label != "relation":
                self.relation_refinements[relation] = label

    def align_entity(self, text: str, context: str = "") -> tuple[OntologyLabel, ...]:
        key = normalize_text(text)
        scores = self._scores(text + " " + context, self.entity_concepts)
        if key in self.entity_refinements:
            scores[self.entity_refinements[key]] = max(
                scores.get(self.entity_refinements[key], 0.0), 0.86
            )
        return self._labels(scores, fallback="entity")

    def align_relation(self, text: str, context: str = "") -> tuple[OntologyLabel, ...]:
        key = normalize_text(text)
        scores = self._scores(text + " " + context, self.relation_concepts)
        if key in self.relation_refinements:
            scores[self.relation_refinements[key]] = max(
                scores.get(self.relation_refinements[key], 0.0), 0.88
            )
        return self._labels(scores, fallback="relation")

    def annotate_fact(self, fact: EvidenceFact) -> EvidenceFact:
        context = f"{fact.subject} {fact.relation} {fact.object} {fact.sentence}"
        return replace(
            fact,
            subject_types=self.align_entity(fact.subject, context),
            relation_types=self.align_relation(fact.relation, context),
            object_types=self.align_entity(fact.object, context),
        )

    def annotate_node(self, node: QueryNode, question: str) -> QueryNode:
        return replace(node, ontology=self.align_entity(node.label, question))

    def annotate_edge(self, edge: QueryEdge, question: str) -> QueryEdge:
        return replace(edge, ontology=self.align_relation(edge.relation, question))

    def save(self, path: str | Path) -> None:
        value = {
            "entity_concepts": self.entity_concepts,
            "relation_concepts": self.relation_concepts,
            "entity_refinements": self.entity_refinements,
            "relation_refinements": self.relation_refinements,
        }
        Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, scorer: SemanticScorer) -> "OntologyRegistry":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = cls(
            scorer,
            entity_concepts={
                key: tuple(rows) for key, rows in value["entity_concepts"].items()
            },
            relation_concepts={
                key: tuple(rows) for key, rows in value["relation_concepts"].items()
            },
        )
        registry.entity_refinements = dict(value.get("entity_refinements", {}))
        registry.relation_refinements = dict(value.get("relation_refinements", {}))
        return registry

    @staticmethod
    def compatibility(
        left: tuple[OntologyLabel, ...], right: tuple[OntologyLabel, ...]
    ) -> float:
        if not left or not right:
            return 0.5
        left_map = {item.name: item.confidence for item in left}
        right_map = {item.name: item.confidence for item in right}
        shared = set(left_map) & set(right_map)
        if not shared:
            return 0.0
        return max(min(left_map[name], right_map[name]) for name in shared)

    def _scores(
        self, text: str, concepts: dict[str, tuple[str, ...]]
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for concept, aliases in concepts.items():
            values = (concept,) + tuple(aliases)
            scores[concept] = max(self.scorer.similarity(text, value) for value in values)
        return scores

    def _best_concept(
        self, text: str, concepts: dict[str, tuple[str, ...]]
    ) -> str:
        scores = self._scores(text, concepts)
        return max(scores, key=scores.get)

    @staticmethod
    def _labels(scores: dict[str, float], fallback: str) -> tuple[OntologyLabel, ...]:
        selected = [
            OntologyLabel(name, score)
            for name, score in sorted(scores.items(), key=lambda row: (-row[1], row[0]))
            if score >= 0.30
        ][:3]
        if not selected:
            selected = [OntologyLabel(fallback, 0.25)]
        return tuple(selected)

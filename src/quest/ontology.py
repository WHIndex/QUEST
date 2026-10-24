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

ARGUMENT_REFINEMENTS = {
    ("authorship", "source"): "work",
    ("authorship", "target"): "author",
    ("birth", "source"): "person",
    ("birth", "target"): "birthplace",
    ("affiliation", "source"): "person",
    ("affiliation", "target"): "organization",
    ("location", "target"): "location",
    ("family", "source"): "person",
    ("family", "target"): "person",
    ("time", "target"): "time",
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
        self.entity_refinement_parents: dict[str, str] = {}
        self.relation_refinement_parents: dict[str, str] = {}

    def induce(self, facts: Iterable[EvidenceFact], min_support: int = 3) -> None:
        facts = list(facts)
        entity_context: dict[str, Counter[str]] = defaultdict(Counter)
        entity_roles: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        entity_sentences: dict[str, Counter[str]] = defaultdict(Counter)
        relation_counts: Counter[str] = Counter()
        relation_parents: dict[str, str] = {}
        for fact in facts:
            relation = normalize_text(fact.relation)
            relation_counts[relation] += 1
            parent = self._best_concept(relation, self.relation_concepts)
            relation_parents[relation] = parent
            entity_context[normalize_text(fact.subject)][relation] += 1
            entity_context[normalize_text(fact.object)][relation] += 1
            entity_roles[normalize_text(fact.subject)][(parent, "source")] += 1
            entity_roles[normalize_text(fact.object)][(parent, "target")] += 1
            entity_sentences[normalize_text(fact.subject)][fact.sentence] += 1
            entity_sentences[normalize_text(fact.object)][fact.sentence] += 1

        for entity, context in entity_context.items():
            if sum(context.values()) < min_support:
                continue
            relation_text = " ".join(relation for relation, _ in context.most_common(4))
            sentence_text = " ".join(
                sentence for sentence, _ in entity_sentences[entity].most_common(2)
            )
            role_text = " ".join(
                f"{role} {family}"
                for (family, role), _ in entity_roles[entity].most_common(3)
            )
            description = f"{entity} {relation_text} {role_text} {sentence_text}"
            label = self._best_concept(description, self.entity_concepts)
            role_label = self._role_refinement(entity_roles[entity])
            refinement = role_label or self._best_alias(description, label)
            if refinement:
                self.entity_refinements[entity] = refinement
                self.entity_refinement_parents[refinement] = label

        for relation, count in relation_counts.items():
            if count < min_support:
                continue
            parent = relation_parents[relation]
            self.relation_refinements[relation] = relation
            self.relation_refinement_parents[relation] = parent

    def align_entity(self, text: str, context: str = "") -> tuple[OntologyLabel, ...]:
        key = normalize_text(text)
        scores = self._scores(text + " " + context, self.entity_concepts)
        for refinement, parent in self.entity_refinement_parents.items():
            score = self.scorer.similarity(text + " " + context, refinement)
            if score >= 0.30:
                scores[refinement] = score
                scores[parent] = max(scores.get(parent, 0.0), score * 0.92)
        if key in self.entity_refinements:
            refinement = self.entity_refinements[key]
            parent = self.entity_refinement_parents.get(refinement, "entity")
            scores[refinement] = max(scores.get(refinement, 0.0), 0.90)
            scores[parent] = max(scores.get(parent, 0.0), 0.86)
        return self._labels(scores, fallback="entity")

    def align_relation(self, text: str, context: str = "") -> tuple[OntologyLabel, ...]:
        key = normalize_text(text)
        scores = self._scores(text + " " + context, self.relation_concepts)
        for refinement, parent in self.relation_refinement_parents.items():
            score = self.scorer.similarity(text + " " + context, refinement)
            if score >= 0.30:
                scores[refinement] = score
                scores[parent] = max(scores.get(parent, 0.0), score * 0.92)
        if key in self.relation_refinements:
            refinement = self.relation_refinements[key]
            parent = self.relation_refinement_parents.get(refinement, "relation")
            scores[refinement] = max(scores.get(refinement, 0.0), 0.92)
            scores[parent] = max(scores.get(parent, 0.0), 0.88)
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
            "entity_refinement_parents": self.entity_refinement_parents,
            "relation_refinement_parents": self.relation_refinement_parents,
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
        registry.entity_refinement_parents = dict(
            value.get("entity_refinement_parents", {})
        )
        registry.relation_refinement_parents = dict(
            value.get("relation_refinement_parents", {})
        )
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

    def _best_alias(self, description: str, parent: str) -> str:
        aliases = self.entity_concepts.get(parent, ())
        if not aliases:
            return ""
        ranked = sorted(
            ((self.scorer.similarity(description, alias), alias) for alias in aliases),
            reverse=True,
        )
        return ranked[0][1] if ranked and ranked[0][0] >= 0.34 else ""

    @staticmethod
    def _role_refinement(roles: Counter[tuple[str, str]]) -> str:
        ranked = sorted(
            (
                (count, ARGUMENT_REFINEMENTS[(family, role)])
                for (family, role), count in roles.items()
                if (family, role) in ARGUMENT_REFINEMENTS
            ),
            reverse=True,
        )
        return ranked[0][1] if ranked else ""

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

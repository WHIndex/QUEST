from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .client import ModelClient
from .model import EvidenceFact, OntologyLabel, Passage
from .ontology import OntologyRegistry
from .semantic import SemanticScorer, normalize_text


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class EvidenceGraph:
    def __init__(
        self,
        passages: Sequence[Passage],
        facts: Sequence[EvidenceFact],
        scorer: SemanticScorer,
    ):
        self.passages = {passage.id: passage for passage in passages}
        self.facts = {fact.id: fact for fact in facts}
        self.scorer = scorer
        self.outgoing: dict[str, list[str]] = defaultdict(list)
        self.incoming: dict[str, list[str]] = defaultdict(list)
        self.entity_facts: dict[str, set[str]] = defaultdict(set)
        self.alias_to_entity: dict[str, str] = {}
        self.relation_cardinality: Counter[str] = Counter()
        self.degree: Counter[str] = Counter()
        self.local_branching: dict[str, float] = {}
        self._entities: dict[str, str] = {}
        self._fact_text: dict[str, str] = {}
        self._build()

    def _build(self) -> None:
        self.outgoing.clear()
        self.incoming.clear()
        self.entity_facts.clear()
        self.alias_to_entity.clear()
        self.relation_cardinality.clear()
        self.degree.clear()
        self.local_branching.clear()
        self._entities.clear()
        self._fact_text.clear()
        for fact in self.facts.values():
            subject = self.entity_key(fact.subject)
            object_ = self.entity_key(fact.object)
            self._entities.setdefault(subject, fact.subject)
            self._entities.setdefault(object_, fact.object)
            self.outgoing[subject].append(fact.id)
            self.incoming[object_].append(fact.id)
            self.entity_facts[subject].add(fact.id)
            self.entity_facts[object_].add(fact.id)
            self.relation_cardinality[normalize_text(fact.relation)] += 1
            self.degree[subject] += 1
            self.degree[object_] += 1
            self._fact_text[fact.id] = (
                f"{fact.subject} {fact.relation} {fact.object}. {fact.sentence}"
            )
            for alias in (fact.subject, fact.object) + tuple(fact.aliases):
                key = normalize_text(alias)
                if key:
                    if self.scorer.similarity(alias, fact.subject) >= self.scorer.similarity(
                        alias, fact.object
                    ):
                        self.alias_to_entity.setdefault(key, subject)
                    else:
                        self.alias_to_entity.setdefault(key, object_)
        for entity in self._entities:
            neighbors = len(self.outgoing[entity]) + len(self.incoming[entity])
            relations = {
                normalize_text(self.facts[fact_id].relation)
                for fact_id in self.entity_facts[entity]
            }
            self.local_branching[entity] = neighbors / max(1, len(relations))

    def add_facts(self, facts: Iterable[EvidenceFact]) -> int:
        added = 0
        for fact in facts:
            if fact.id in self.facts or fact.passage_id not in self.passages:
                continue
            self.facts[fact.id] = fact
            added += 1
        if added:
            self._build()
        return added

    @staticmethod
    def entity_key(value: str) -> str:
        return normalize_text(value)

    def resolve_entity(self, value: str) -> str | None:
        key = normalize_text(value)
        if key in self._entities:
            return key
        if key in self.alias_to_entity:
            return self.alias_to_entity[key]
        ranked = self.scorer.rank(value, list(self._entities.values()))
        if ranked and ranked[0][1] >= 0.72:
            return list(self._entities)[ranked[0][0]]
        return None

    def entity_label(self, entity_key: str) -> str:
        return self._entities.get(entity_key, entity_key)

    def adjacent(
        self, entity_key: str, *, direction: str, allowed: set[str] | None = None
    ) -> list[EvidenceFact]:
        fact_ids = self.outgoing[entity_key] if direction == "out" else self.incoming[entity_key]
        if allowed is not None:
            fact_ids = [fact_id for fact_id in fact_ids if fact_id in allowed]
        return [self.facts[fact_id] for fact_id in fact_ids]

    def rank_facts(
        self,
        text: str,
        limit: int,
        *,
        allowed: set[str] | None = None,
    ) -> list[tuple[EvidenceFact, float]]:
        ids = list(allowed if allowed is not None else self.facts)
        ranked = self.scorer.rank(text, [self._fact_text[fact_id] for fact_id in ids])
        return [(self.facts[ids[index]], score) for index, score in ranked[:limit]]

    def rank_passages(self, text: str, limit: int) -> list[tuple[Passage, float]]:
        rows = list(self.passages.values())
        ranked = self.scorer.rank(
            text, [f"{passage.title} {passage.text}" for passage in rows]
        )
        return [(rows[index], score) for index, score in ranked[:limit]]

    def direct_cardinality(self, relation: str) -> int:
        exact = self.relation_cardinality.get(normalize_text(relation), 0)
        if exact:
            return exact
        ranked = self.scorer.rank(relation, list(self.relation_cardinality))
        return sum(
            self.relation_cardinality[list(self.relation_cardinality)[index]]
            for index, score in ranked[:4]
            if score >= 0.45
        )

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "passages.jsonl").open("w", encoding="utf-8") as handle:
            for passage in self.passages.values():
                handle.write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")
        with (target / "facts.jsonl").open("w", encoding="utf-8") as handle:
            for fact in self.facts.values():
                row = asdict(fact)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path, scorer: SemanticScorer) -> "EvidenceGraph":
        source = Path(directory)
        passages = [
            Passage(**row) for row in _read_jsonl(source / "passages.jsonl")
        ]
        facts = []
        for row in _read_jsonl(source / "facts.jsonl"):
            row["subject_types"] = tuple(OntologyLabel(**value) for value in row.get("subject_types", []))
            row["relation_types"] = tuple(OntologyLabel(**value) for value in row.get("relation_types", []))
            row["object_types"] = tuple(OntologyLabel(**value) for value in row.get("object_types", []))
            row["aliases"] = tuple(row.get("aliases", []))
            facts.append(EvidenceFact(**row))
        return cls(passages, facts, scorer)


class CorpusBuilder:
    def __init__(
        self,
        client: ModelClient,
        ontology: OntologyRegistry,
        scorer: SemanticScorer,
    ):
        self.client = client
        self.ontology = ontology
        self.scorer = scorer

    def build(self, passages: Sequence[Passage]) -> EvidenceGraph:
        raw: list[EvidenceFact] = []
        for passage in passages:
            raw.extend(self.extract(passage))
        normalized = self._normalize_entities(raw)
        self.ontology.induce(normalized)
        annotated = [self.ontology.annotate_fact(fact) for fact in normalized]
        return EvidenceGraph(passages, annotated, self.scorer)

    def extract(self, passage: Passage) -> list[EvidenceFact]:
        response = self.client.complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract factual directed relations from the supplied passage. "
                        "Keep distinct relations separate. Return an object with a facts array. "
                        "Each fact has subject, relation, object, and one exact supporting sentence. "
                        "Do not emit a fact that the passage does not support."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"title": passage.title, "text": passage.text}, ensure_ascii=False
                    ),
                },
            ]
        )
        sentences = [row.strip() for row in _SENTENCE.split(passage.text) if row.strip()]
        facts: list[EvidenceFact] = []
        for row in response.get("facts", []):
            if not isinstance(row, dict):
                continue
            subject = str(row.get("subject", "")).strip()
            relation = str(row.get("relation", "")).strip()
            object_ = str(row.get("object", "")).strip()
            sentence = str(row.get("sentence", "")).strip()
            if not subject or not relation or not object_:
                continue
            if sentence not in passage.text:
                sentence = max(
                    sentences,
                    key=lambda value: self.scorer.similarity(
                        f"{subject} {relation} {object_}", value
                    ),
                    default=passage.text,
                )
            digest = hashlib.sha1(
                f"{passage.id}\0{subject}\0{relation}\0{object_}\0{sentence}".encode()
            ).hexdigest()[:16]
            facts.append(
                EvidenceFact(
                    id=f"f_{digest}",
                    subject=subject,
                    relation=relation,
                    object=object_,
                    passage_id=passage.id,
                    sentence=sentence,
                )
            )
        return facts

    def _normalize_entities(self, facts: Sequence[EvidenceFact]) -> list[EvidenceFact]:
        canonical: list[str] = []
        aliases: dict[str, set[str]] = defaultdict(set)

        def resolve(value: str) -> str:
            for existing in canonical:
                if self.scorer.similarity(value, existing) >= 0.88:
                    aliases[existing].add(value)
                    return existing
            canonical.append(value)
            return value

        rows = []
        for fact in facts:
            subject = resolve(fact.subject)
            object_ = resolve(fact.object)
            rows.append(replace(fact, subject=subject, object=object_))
        return [
            replace(
                fact,
                aliases=tuple(
                    sorted((aliases[fact.subject] | aliases[fact.object]) - {fact.subject, fact.object})
                ),
            )
            for fact in rows
        ]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value

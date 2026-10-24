from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .client import ModelClient
from .model import NodeKind, QueryEdge, QueryGraph, QueryNode, QueryOperator
from .ontology import OntologyRegistry


@dataclass(frozen=True)
class CriticIssue:
    owner: str
    code: str
    target: str
    message: str


class DecomposerAgent:
    def __init__(self, client: ModelClient):
        self.client = client

    def decompose(
        self,
        question: str,
        previous: dict[str, Any] | None = None,
        issues: list[CriticIssue] | None = None,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Decompose a question into the smallest factual constraints needed to answer it. "
                    "Return one canonical intermediate representation with entities, constraints, "
                    "answer_nodes, and operators. Each entity has id, label, and kind, where kind is "
                    "entity, variable, or value. Each constraint has id, source, relation, target, "
                    "and required. Reuse exactly the same variable id across constraints when the "
                    "question requires the same unknown entity. Do not invent evidence entities. "
                    "The answer projection must name existing entity ids."
                ),
            },
            {"role": "user", "content": question},
        ]
        if previous is not None and issues:
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current": previous,
                            "issues": [issue.__dict__ for issue in issues],
                            "instruction": "Repair only the stated decomposition or answer-projection errors.",
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        return self.client.complete_json(messages)


class AlignmentAgent:
    def __init__(self, client: ModelClient, ontology: OntologyRegistry):
        self.client = client
        self.ontology = ontology

    def align(
        self,
        question: str,
        intermediate: dict[str, Any],
        previous: dict[str, Any] | None = None,
        issues: list[CriticIssue] | None = None,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Normalize the supplied query representation without changing its logical intent. "
                    "Preserve all entity ids, shared variable ids, constraints, and answer nodes. "
                    "Resolve entity references and coreference, normalize relation wording, and make "
                    "every relation direction explicit from source to target. Return entities, "
                    "constraints, answer_nodes, and operators. Do not add unsupported constraints."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "intermediate": intermediate},
                    ensure_ascii=False,
                ),
            },
        ]
        if previous is not None and issues:
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current": previous,
                            "issues": [issue.__dict__ for issue in issues],
                            "instruction": "Repair only alignment, direction, ontology, or variable correspondence errors.",
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        aligned = self.client.complete_json(messages)
        return self._attach_ontology(question, aligned)

    def _attach_ontology(
        self, question: str, intermediate: dict[str, Any]
    ) -> dict[str, Any]:
        value = deepcopy(intermediate)
        for node in value.get("entities", []):
            labels = self.ontology.align_entity(str(node.get("label", "")), question)
            node["ontology"] = [item.__dict__ for item in labels]
        for edge in value.get("constraints", []):
            labels = self.ontology.align_relation(str(edge.get("relation", "")), question)
            edge["ontology"] = [item.__dict__ for item in labels]
        return value


class CriticAgent:
    def __init__(self, client: ModelClient):
        self.client = client

    def review(self, question: str, graph: QueryGraph) -> list[CriticIssue]:
        response = self.client.complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Review whether one query graph faithfully captures its question. Check that "
                        "all required atomic constraints are present, relation meanings and directions "
                        "are correct, shared variables denote the same entity across constraints, and "
                        "the answer projection requests the intended information. Return an issues "
                        "array. Each issue has owner, code, target, and message. Use owner decomposer "
                        "for missing or wrongly decomposed constraints and answer projection errors. "
                        "Use owner alignment for entity, relation, direction, ontology, coreference, "
                        "or variable correspondence errors. Return an empty array when correct."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "graph": graph_to_dict(graph)},
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        issues = []
        for row in response.get("issues", []):
            if not isinstance(row, dict):
                continue
            owner = str(row.get("owner", "alignment")).strip().lower()
            if owner not in {"decomposer", "alignment"}:
                owner = "alignment"
            message = str(row.get("message", "")).strip()
            if not message:
                continue
            issues.append(
                CriticIssue(
                    owner=owner,
                    code=str(row.get("code", "semantic_error")),
                    target=str(row.get("target", "graph")),
                    message=message,
                )
            )
        return issues


class QueryCompiler:
    def __init__(
        self,
        decomposer: DecomposerAgent,
        aligner: AlignmentAgent,
        critic: CriticAgent,
        repair_budget: int = 2,
    ):
        self.decomposer = decomposer
        self.aligner = aligner
        self.critic = critic
        self.repair_budget = repair_budget

    def compile(self, question: str) -> QueryGraph:
        decomposed = self.decomposer.decompose(question)
        aligned = self.aligner.align(question, decomposed)
        issues: list[CriticIssue] = []
        graph: QueryGraph | None = None

        for attempt in range(self.repair_budget + 1):
            try:
                graph = assemble_query_graph(aligned)
                issues = self._all_issues(question, graph)
            except (KeyError, TypeError, ValueError) as error:
                graph = None
                issues = [
                    CriticIssue(
                        "alignment",
                        "invalid_graph",
                        "graph",
                        str(error),
                    )
                ]
            if not issues and graph is not None:
                return graph
            if attempt == self.repair_budget:
                break
            decomposition_issues = [
                issue for issue in issues if issue.owner == "decomposer"
            ]
            alignment_issues = [issue for issue in issues if issue.owner == "alignment"]
            if decomposition_issues:
                decomposed = self.decomposer.decompose(
                    question, decomposed, decomposition_issues
                )
                aligned = self.aligner.align(question, decomposed)
            elif alignment_issues:
                aligned = self.aligner.align(
                    question, decomposed, aligned, alignment_issues
                )
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise ValueError(f"query graph did not pass semantic review: {detail}")

    def _all_issues(self, question: str, graph: QueryGraph) -> list[CriticIssue]:
        issues = validate_query_graph(graph)
        if issues:
            return issues
        return self.critic.review(question, graph)


def assemble_query_graph(value: dict[str, Any]) -> QueryGraph:
    nodes = []
    for row in value.get("entities", []):
        if not isinstance(row, dict):
            continue
        kind_text = str(row.get("kind", "variable")).lower()
        kind = NodeKind(kind_text) if kind_text in NodeKind._value2member_map_ else NodeKind.VARIABLE
        ontology = tuple(
            _ontology_label(item) for item in row.get("ontology", []) if isinstance(item, dict)
        )
        nodes.append(
            QueryNode(
                id=str(row.get("id", "")).strip(),
                label=str(row.get("label", "")).strip(),
                kind=kind,
                ontology=ontology,
                grounded_id=(
                    str(row.get("grounded_id")).strip()
                    if row.get("grounded_id") is not None
                    else None
                ),
            )
        )

    edges = []
    for index, row in enumerate(value.get("constraints", [])):
        if not isinstance(row, dict):
            continue
        ontology = tuple(
            _ontology_label(item) for item in row.get("ontology", []) if isinstance(item, dict)
        )
        edges.append(
            QueryEdge(
                id=str(row.get("id") or f"e{index + 1}"),
                source=str(row.get("source", "")).strip(),
                relation=str(row.get("relation", "")).strip(),
                target=str(row.get("target", "")).strip(),
                ontology=ontology,
                required=_as_bool(row.get("required", True)),
            )
        )

    operators = []
    for row in value.get("operators", []):
        if not isinstance(row, dict):
            continue
        arguments = row.get("arguments", [])
        if isinstance(arguments, str):
            arguments = [arguments]
        operators.append(
            QueryOperator(
                name=str(row.get("name", "")).strip(),
                arguments=tuple(str(item) for item in arguments),
                value=row.get("value"),
            )
        )

    answer_nodes = value.get("answer_nodes", [])
    if isinstance(answer_nodes, str):
        answer_nodes = [answer_nodes]
    return QueryGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        answer_nodes=tuple(str(item) for item in answer_nodes),
        operators=tuple(operators),
    )


def validate_query_graph(graph: QueryGraph) -> list[CriticIssue]:
    issues: list[CriticIssue] = []
    node_map = graph.node_map
    if not graph.required_edges:
        issues.append(
            CriticIssue("decomposer", "missing_constraint", "graph", "no required relation constraint")
        )
    if not graph.answer_nodes:
        issues.append(
            CriticIssue("decomposer", "missing_projection", "graph", "answer projection is empty")
        )
    for node in graph.nodes:
        if not node.id or not node.label:
            issues.append(
                CriticIssue("alignment", "invalid_node", node.id or "node", "node id and label are required")
            )
    for edge in graph.edges:
        if not edge.relation:
            issues.append(
                CriticIssue("alignment", "missing_relation", edge.id, "relation is empty")
            )
        if edge.source == edge.target:
            issues.append(
                CriticIssue("alignment", "self_relation", edge.id, "relation endpoints must be distinct")
            )
        if edge.source not in node_map or edge.target not in node_map:
            issues.append(
                CriticIssue("alignment", "unknown_endpoint", edge.id, "constraint references an unknown node")
            )
    return issues


def graph_to_dict(graph: QueryGraph) -> dict[str, Any]:
    return {
        "entities": [
            {
                "id": node.id,
                "label": node.label,
                "kind": node.kind.value,
                "grounded_id": node.grounded_id,
                "ontology": [item.__dict__ for item in node.ontology],
            }
            for node in graph.nodes
        ],
        "constraints": [
            {
                "id": edge.id,
                "source": edge.source,
                "relation": edge.relation,
                "target": edge.target,
                "required": edge.required,
                "ontology": [item.__dict__ for item in edge.ontology],
            }
            for edge in graph.edges
        ],
        "answer_nodes": list(graph.answer_nodes),
        "operators": [
            {"name": item.name, "arguments": list(item.arguments), "value": item.value}
            for item in graph.operators
        ],
    }


def _ontology_label(value: dict[str, Any]):
    from .model import OntologyLabel

    return OntologyLabel(
        name=str(value.get("name", "")).strip(),
        confidence=max(0.0, min(1.0, float(value.get("confidence", 0.0)))),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .client import EmbeddingClient, ModelClient
from .evidence import CorpusBuilder, EvidenceGraph
from .io import answer_to_dict, append_jsonl, load_passages, read_jsonl
from .ontology import OntologyRegistry
from .pipeline import QUEST
from .semantic import SemanticScorer


def main() -> None:
    parser = argparse.ArgumentParser(prog="quest")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--passages", required=True)
    build.add_argument("--index", required=True)

    ask = commands.add_parser("ask")
    ask.add_argument("--index", required=True)
    ask.add_argument("--question", required=True)

    run = commands.add_parser("run")
    run.add_argument("--index", required=True)
    run.add_argument("--questions", required=True)
    run.add_argument("--output", required=True)

    arguments = parser.parse_args()
    client = ModelClient()
    scorer = _scorer()

    if arguments.command == "build":
        ontology = OntologyRegistry(scorer)
        builder = CorpusBuilder(client, ontology, scorer)
        graph = builder.build(load_passages(arguments.passages))
        graph.save(arguments.index)
        ontology.save(Path(arguments.index) / "ontology.json")
        print(json.dumps({"passages": len(graph.passages), "facts": len(graph.facts)}))
        return

    ontology = OntologyRegistry.load(
        Path(arguments.index) / "ontology.json", scorer
    )
    graph = EvidenceGraph.load(arguments.index, scorer)
    system = QUEST(graph, ontology, scorer, client)

    if arguments.command == "ask":
        result = system.answer(arguments.question)
        print(json.dumps(answer_to_dict(result), ensure_ascii=False, indent=2))
        return

    output = Path(arguments.output)
    completed = set()
    if output.exists():
        completed = {str(row.get("id")) for row in read_jsonl(output)}
    for index, row in enumerate(read_jsonl(arguments.questions)):
        row_id = str(row.get("id", index))
        if row_id in completed:
            continue
        result = system.answer(str(row["question"]))
        append_jsonl(
            output,
            {"id": row_id, "question": row["question"], **answer_to_dict(result)},
        )


def _scorer() -> SemanticScorer:
    required = ("EMBED_BASE_URL", "EMBED_API_KEY", "EMBED_MODEL")
    if all(os.environ.get(name) for name in required):
        return SemanticScorer(EmbeddingClient())
    return SemanticScorer()


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .certificate import CertificateChecker
from .client import ModelClient
from .completion import EvidenceCompletion, RetrievalAgent
from .config import PipelineConfig
from .context import ContextSelection, ContextSelector
from .dp import BindDP, DPResult
from .evidence import CorpusBuilder, EvidenceGraph
from .matching import CostGuidedBindingSearch, RealizationGenerator, SearchResult
from .model import CertifiedMatch, QueryGraph
from .ontology import OntologyRegistry
from .query import AlignmentAgent, CriticAgent, DecomposerAgent, QueryCompiler, graph_to_dict
from .semantic import SemanticScorer


@dataclass(frozen=True)
class AnswerResult:
    answer: str | None
    query_graph: QueryGraph
    certified: tuple[CertifiedMatch, ...]
    context: ContextSelection
    diagnostics: dict[str, Any]


class Reader:
    def __init__(self, client: ModelClient):
        self.client = client

    def answer(self, question: str, context: ContextSelection) -> str:
        evidence = "\n\n".join(
            f"[{passage.id}] {passage.title}\n{passage.text}"
            for passage in context.passages
        )
        return self.client.complete_text(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the supplied source passages. "
                        "Give a concise answer. If the passages do not support an answer, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nSource passages:\n{evidence}",
                },
            ],
            max_tokens=512,
        ).strip()


class QUEST:
    def __init__(
        self,
        graph: EvidenceGraph,
        ontology: OntologyRegistry,
        scorer: SemanticScorer,
        client: ModelClient,
        config: PipelineConfig | None = None,
    ):
        self.graph = graph
        self.ontology = ontology
        self.scorer = scorer
        self.client = client
        self.config = config or PipelineConfig()

        self.builder = CorpusBuilder(client, ontology, scorer)
        self.compiler = QueryCompiler(
            DecomposerAgent(client),
            AlignmentAgent(client, ontology),
            CriticAgent(client),
            self.config.compiler_repairs,
        )
        self.generator = RealizationGenerator(
            graph, ontology, scorer, self.config.search
        )
        self.search = CostGuidedBindingSearch(
            graph, self.generator, self.config.search
        )
        self.dp = BindDP(self.config.search)
        self.checker = CertificateChecker(graph, scorer)
        self.completion = EvidenceCompletion(
            graph,
            ontology,
            scorer,
            self.builder,
            RetrievalAgent(client),
            self.generator,
            self.dp,
            self.checker,
            self.config.search,
        )
        self.selector = ContextSelector(
            graph,
            self.config.context_passages,
            self.config.context_characters,
        )
        self.reader = Reader(client)

    def retrieve(
        self, question: str, query_graph: QueryGraph | None = None
    ) -> tuple[
        QueryGraph,
        SearchResult,
        DPResult,
        tuple[CertifiedMatch, ...],
        dict[str, Any],
    ]:
        query = query_graph or self.compiler.compile(question)
        search = self.search.run(query)
        dp_result = self.dp.run(query, search.active)
        certified = tuple(
            self.checker.certify_all(
                query, dp_result.terminal, self.config.search.match_budget
            )
        )
        recovery_data: dict[str, Any] = {
            "rounds": 0,
            "requests": 0,
            "added_realizations": 0,
            "reused_subsets": 0,
            "recomputed_subsets": 0,
        }
        if not certified:
            recovery = self.completion.recover(
                question, query, search, dp_result
            )
            search = recovery.search
            dp_result = recovery.dp
            certified = recovery.certified
            recovery_data = {
                "rounds": recovery.rounds,
                "requests": recovery.requests,
                "added_realizations": recovery.added_realizations,
                "reused_subsets": recovery.reused_subsets,
                "recomputed_subsets": recovery.recomputed_subsets,
            }
        return query, search, dp_result, certified, recovery_data

    def answer(
        self, question: str, query_graph: QueryGraph | None = None
    ) -> AnswerResult:
        query, search, dp_result, certified, recovery = self.retrieve(
            question, query_graph
        )
        context = self.selector.select(query, certified)
        answer = self.reader.answer(question, context) if certified and context.passages else None
        diagnostics = {
            "query_graph": graph_to_dict(query),
            "edge_order": search.edge_order,
            "operations": search.operations,
            "cached_realizations": {
                edge_id: len(rows) for edge_id, rows in search.cached.items()
            },
            "active_realizations": {
                edge_id: len(rows) for edge_id, rows in search.active.items()
            },
            "terminal_states": len(dp_result.terminal),
            "certified_matches": len(certified),
            "recovery": recovery,
        }
        return AnswerResult(answer, query, certified, context, diagnostics)

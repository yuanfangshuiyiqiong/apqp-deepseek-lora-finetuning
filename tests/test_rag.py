import json
import tempfile
import unittest
from pathlib import Path

from back.rag.embeddings import HashingEmbeddingModel
from back.rag.prompting import build_rag_prompt
from back.rag.vector_store import JsonlVectorStore, VectorRecord, chunk_text


class RagTests(unittest.TestCase):
    def test_chunk_text_splits_with_overlap(self):
        chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=1)

        self.assertEqual(chunks, ["abcd", "defg", "ghij"])

    def test_hashing_embedding_is_normalized_and_deterministic(self):
        embedder = HashingEmbeddingModel(dimension=16)

        first = embedder.embed("APQP risk management")
        second = embedder.embed("APQP risk management")

        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first) ** 0.5, 1.0)

    def test_jsonl_vector_store_returns_most_relevant_document(self):
        embedder = HashingEmbeddingModel(dimension=32)
        records = [
            {
                "id": "risk",
                "text": "APQP risk review should identify preventive actions.",
                "embedding": embedder.embed("APQP risk review should identify preventive actions."),
                "metadata": {"source": "risk.md", "chunk_index": 1},
            },
            {
                "id": "schedule",
                "text": "Project schedule tracking focuses on milestones and owners.",
                "embedding": embedder.embed("Project schedule tracking focuses on milestones and owners."),
                "metadata": {"source": "schedule.md", "chunk_index": 1},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vector_store.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            store = JsonlVectorStore.load(path)
            results = store.search(embedder.embed("How should we review APQP risk?"), top_k=1)

        self.assertEqual(results[0].record.id, "risk")
        self.assertGreater(results[0].score, 0)

    def test_short_keyword_question_matches_relevant_apqp_document(self):
        embedder = HashingEmbeddingModel()
        doc_text = (
            "APQP 是产品质量先期策划方法，通常用于新产品开发过程中的质量策划、"
            "风险识别、过程控制和量产准备。"
        )
        store = JsonlVectorStore(
            [
                VectorRecord(
                    id="apqp",
                    text=doc_text,
                    embedding=embedder.embed(doc_text),
                    metadata={"source": "apqp.md", "chunk_index": 1},
                )
            ]
        )

        results = store.search(embedder.embed("APQP 是什么？"), top_k=1, min_score=0.05)

        self.assertEqual(results[0].record.id, "apqp")

    def test_build_rag_prompt_injects_sources_and_no_evidence_rule(self):
        prompt = build_rag_prompt(
            user_question="APQP risk review?",
            contexts=[
                {
                    "text": "Risk review should happen in early planning.",
                    "metadata": {"source": "risk.md", "chunk_index": 2},
                    "score": 0.88,
                }
            ],
        )

        self.assertIn("Risk review should happen in early planning.", prompt)
        self.assertIn("risk.md", prompt)
        self.assertIn("只有在未检索到相关资料时", prompt)


if __name__ == "__main__":
    unittest.main()

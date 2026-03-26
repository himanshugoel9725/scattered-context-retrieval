"""ScatterQA benchmark builder pipeline.

Orchestrates: download novels → NER/coref → entity annotation → QA generation → validation.
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.data.gutenberg import download_gutenberg_text, split_into_chapters, SELECTED_NOVELS
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.coref_resolver import resolve_coreferences
from src.evaluation.gold_schema import GoldEvidence, QueryMetadata, save_gold_evidence
from src.utils.config import data_dir

logger = logging.getLogger(__name__)


def _novel_tuple_to_dict(t: tuple) -> dict:
    """Convert SELECTED_NOVELS tuple to dict."""
    return {"book_id": t[0], "title": t[1], "author": t[2]}


class ScatterQABuilder:
    """Build the ScatterQA benchmark from Project Gutenberg novels."""

    def __init__(self, n_novels: int = 50, entities_per_novel: int = 10,
                 questions_per_entity: int = 5, output_dir: str | None = None):
        self.n_novels = n_novels
        self.entities_per_novel = entities_per_novel
        self.questions_per_entity = questions_per_entity
        self.output_dir = Path(output_dir) if output_dir else data_dir("scatterqa")
        self.chunker = Chunker(chunk_size=512, overlap=128)

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "build_checkpoint.jsonl"

    def _load_checkpoint(self) -> tuple[list[GoldEvidence], set[int]]:
        """Load checkpoint if exists. Returns (records, completed_book_ids)."""
        cp = self._checkpoint_path()
        if not cp.exists():
            return [], set()
        from src.evaluation.gold_schema import load_gold_evidence
        records = load_gold_evidence(cp)
        # Extract book IDs from entity_id format "gutenberg_{book_id}:..."
        done_ids = set()
        for r in records:
            parts = r.entity_id.split(":")
            if parts[0].startswith("gutenberg_"):
                try:
                    done_ids.add(int(parts[0].replace("gutenberg_", "")))
                except ValueError:
                    pass
        return records, done_ids

    def _save_checkpoint(self, records: list[GoldEvidence]):
        save_gold_evidence(records, self._checkpoint_path())

    def build(self):
        """Run the full pipeline with checkpoint/resume support."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        novels = [_novel_tuple_to_dict(t) for t in SELECTED_NOVELS[:self.n_novels]]

        all_records, done_ids = self._load_checkpoint()
        if done_ids:
            logger.info("Resuming from checkpoint: %d novels done, %d records",
                        len(done_ids), len(all_records))

        for novel_info in novels:
            if novel_info["book_id"] in done_ids:
                logger.info("Skipping (already done): %s", novel_info["title"])
                continue
            logger.info("Processing: %s by %s", novel_info["title"], novel_info["author"])
            try:
                records = self._process_novel(novel_info)
                all_records.extend(records)
                self._save_checkpoint(all_records)
                logger.info("Checkpoint saved: %d total records", len(all_records))
            except Exception as e:
                logger.error("Failed on %s: %s", novel_info["title"], e)
                continue

        # Save final outputs
        save_gold_evidence(all_records, self.output_dir / "gold_evidence.jsonl")

        # Split dev/test (20/80)
        n_dev = max(1, len(all_records) // 5)
        dev_records = all_records[:n_dev]
        test_records = all_records[n_dev:]
        save_gold_evidence(dev_records, self.output_dir / "dev.jsonl")
        save_gold_evidence(test_records, self.output_dir / "test.jsonl")

        logger.info("ScatterQA built: %d total records (%d dev, %d test)",
                     len(all_records), len(dev_records), len(test_records))
        return all_records

    def _process_novel(self, novel_info: dict) -> list[GoldEvidence]:
        """Process a single novel: download, chunk, NER, annotate, generate QA."""
        book_id = novel_info["book_id"]

        # Download (caches to data/raw/gutenberg/)
        text = download_gutenberg_text(book_id)
        text = clean_text(text)
        doc_id = f"gutenberg_{book_id}"

        # Chunk
        chunks = self.chunker.chunk_document(doc_id, text)
        chunk_map = {c.chunk_id: c for c in chunks}

        # Build entity index
        entity_idx = EntityIndex(doc_id)
        entity_idx.build_from_chunks(chunks)

        # Coref resolution (optional — expensive on CPU)
        try:
            coref_result = resolve_coreferences(text[:50000])
            entity_idx.merge_coreferences(coref_result["clusters"])
        except Exception as e:
            logger.warning("Coref failed for %s: %s", doc_id, e)

        # Select top entities
        top_entities = entity_idx.get_top_entities(self.entities_per_novel)

        # Generate QA pairs for each entity
        records = []
        from src.benchmark.qa_generator import generate_entity_questions
        for entity_data in top_entities:
            try:
                entity_records = generate_entity_questions(
                    entity_data=entity_data,
                    chunk_map=chunk_map,
                    doc_id=doc_id,
                    novel_info=novel_info,
                    n_questions=self.questions_per_entity,
                )
                records.extend(entity_records)
            except Exception as e:
                logger.warning("QA gen failed for entity %s: %s",
                               entity_data.get("canonical", "?"), e)
        return records

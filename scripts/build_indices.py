"""Build vector + entity indices for all downloaded datasets."""

import argparse
import logging

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.indexing.coref_resolver import resolve_coreferences

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["narrativeqa", "qasper", "cuad"],
                        help="Datasets to index")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--coref", action="store_true", help="Run coreference resolution")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    chunker = Chunker(chunk_size=args.chunk_size, overlap=args.overlap)

    for ds_name in args.datasets:
        logger.info("=== Building indices for %s ===", ds_name)
        dataset = load_dataset(ds_name)
        if not dataset:
            logger.warning("Skipping %s", ds_name)
            continue

        docs = dataset[:args.max_docs] if args.max_docs else dataset
        all_chunks = []

        for doc in docs:
            text = clean_text(doc.text)
            chunks = chunker.chunk_document(doc.doc_id, text)
            all_chunks.extend(chunks)

            # Entity index per document
            entity_idx = EntityIndex(doc.doc_id)
            entity_idx.build_from_chunks(chunks)

            if args.coref:
                try:
                    coref = resolve_coreferences(text[:50000])
                    entity_idx.merge_coreferences(coref["clusters"])
                except Exception as e:
                    logger.warning("Coref failed for %s: %s", doc.doc_id, e)

            entity_idx.save()
            logger.info("Doc %s: %d chunks, %d entities",
                        doc.doc_id, len(chunks), len(entity_idx.entities))

        # Build combined vector index for the dataset
        if all_chunks:
            vector_idx = VectorIndex()
            vector_idx.add(
                [c.chunk_id for c in all_chunks],
                [c.text for c in all_chunks],
            )
            vector_idx.save(ds_name)
            logger.info("Dataset %s: %d total chunks indexed", ds_name, len(all_chunks))


if __name__ == "__main__":
    main()

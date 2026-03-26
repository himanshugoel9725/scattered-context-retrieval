"""Build the ScatterQA benchmark."""

import argparse
import logging

from src.benchmark.scatterqa_builder import ScatterQABuilder

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-novels", type=int, default=50)
    parser.add_argument("--entities-per-novel", type=int, default=10)
    parser.add_argument("--questions-per-entity", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    builder = ScatterQABuilder(
        n_novels=args.n_novels,
        entities_per_novel=args.entities_per_novel,
        questions_per_entity=args.questions_per_entity,
    )
    records = builder.build()
    logger.info("ScatterQA built: %d total QA records", len(records))


if __name__ == "__main__":
    main()

"""Download all datasets to data/raw/."""

import argparse
import logging

from src.data.loaders import load_dataset
from src.data.gutenberg import download_gutenberg_text, SELECTED_NOVELS
from src.utils.config import data_dir

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["narrativeqa", "qasper", "cuad", "quality"],
                        help="Datasets to download")
    parser.add_argument("--gutenberg", action="store_true", help="Download Gutenberg novels")
    parser.add_argument("--n-novels", type=int, default=50, help="Number of Gutenberg novels")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for ds in args.datasets:
        logger.info("Downloading %s...", ds)
        try:
            dataset = load_dataset(ds)
            if dataset:
                logger.info("%s: loaded %d documents", ds, len(dataset))
            else:
                logger.warning("%s: load returned empty", ds)
        except Exception as e:
            logger.error("Failed to download %s: %s", ds, e)

    if args.gutenberg:
        save_dir = str(data_dir("raw/gutenberg"))
        novels = SELECTED_NOVELS[:args.n_novels]
        for i, novel in enumerate(novels):
            logger.info("[%d/%d] Downloading: %s", i + 1, len(novels), novel["title"])
            try:
                download_gutenberg_text(novel["book_id"], save_dir=save_dir)
            except Exception as e:
                logger.error("Failed: %s — %s", novel["title"], e)


if __name__ == "__main__":
    main()

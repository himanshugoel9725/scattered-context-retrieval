"""Dataset loaders with direct download fallbacks — no HuggingFace dependency required.

Each loader returns a list of Document dicts:
  {"doc_id": str, "text": str, "questions": [{"q_id", "question", "answer"}]}
"""

import csv
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.utils.config import data_dir

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    q_id: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class Document:
    doc_id: str
    text: str
    questions: list[QAPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with progress bar if not already cached."""
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", desc or url, dest)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


# ============================================================
# NarrativeQA — direct download from GitHub
# ============================================================

_NARRATIVEQA_BASE = "https://raw.githubusercontent.com/deepmind/narrativeqa/master"


def load_narrativeqa(max_docs: int | None = None) -> list[Document]:
    """Load NarrativeQA from direct GitHub download.

    Downloads qaps.csv (QA pairs) and documents.csv (document metadata).
    Note: Full document texts require separate download of source books/scripts.
    For our purposes, we use the summaries as document text for prototyping,
    and the full texts when available from Project Gutenberg.
    """
    raw = data_dir("raw/narrativeqa")

    # Download QA pairs and document list
    qaps_path = _download_file(
        f"{_NARRATIVEQA_BASE}/qaps.csv", raw / "qaps.csv", "NarrativeQA QA pairs"
    )
    docs_path = _download_file(
        f"{_NARRATIVEQA_BASE}/documents.csv", raw / "documents.csv", "NarrativeQA docs"
    )
    summaries_path = _download_file(
        f"{_NARRATIVEQA_BASE}/third_party/wikipedia/summaries.csv",
        raw / "summaries.csv", "NarrativeQA summaries"
    )

    # Parse documents
    doc_map: dict[str, dict[str, str]] = {}
    with open(docs_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doc_map[row["document_id"]] = {
                "kind": row.get("kind", ""),
                "url": row.get("document_url", ""),
            }

    # Parse summaries
    summary_map: dict[str, str] = {}
    with open(summaries_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            summary_map[row["document_id"]] = row.get("summary", "")

    # Parse QA pairs and group by document
    qa_by_doc: dict[str, list[QAPair]] = {}
    with open(qaps_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["document_id"]
            if doc_id not in qa_by_doc:
                qa_by_doc[doc_id] = []
            qa_by_doc[doc_id].append(QAPair(
                q_id=f"nqa_{doc_id}_{len(qa_by_doc[doc_id])}",
                question=row.get("question", ""),
                answer=row.get("answer1", ""),
                metadata={"answer2": row.get("answer2", ""), "set": row.get("set", "")},
            ))

    # Build Document objects
    documents = []
    for doc_id, qas in qa_by_doc.items():
        text = summary_map.get(doc_id, "")
        if not text:
            continue
        documents.append(Document(
            doc_id=doc_id,
            text=text,
            questions=qas,
            metadata=doc_map.get(doc_id, {}),
        ))
        if max_docs and len(documents) >= max_docs:
            break

    logger.info("Loaded %d NarrativeQA documents with %d total QA pairs",
                len(documents), sum(len(d.questions) for d in documents))
    return documents


# ============================================================
# QASPER — direct download from AllenAI
# ============================================================

_QASPER_TGZ_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"


def load_qasper(split: str = "dev", max_docs: int | None = None) -> list[Document]:
    """Load QASPER from direct download."""
    import tarfile
    raw = data_dir("raw/qasper")
    json_path = raw / f"qasper-{split}-v0.3.json"

    if not json_path.exists():
        tgz_path = _download_file(_QASPER_TGZ_URL, raw / "qasper-train-dev-v0.3.tgz", "QASPER train+dev")
        with tarfile.open(tgz_path, "r:gz") as tf:
            tf.extractall(raw, filter="data")
        logger.info("Extracted QASPER archive to %s", raw)

    path = json_path

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for paper_id, paper in data.items():
        # Build full text from sections
        sections = []
        for sec in paper.get("full_text", []):
            section_name = sec.get("section_name", "")
            paragraphs = sec.get("paragraphs", [])
            if section_name:
                sections.append(f"## {section_name}")
            sections.extend(paragraphs)
        full_text = "\n\n".join(sections)

        # Extract QA pairs
        qas = []
        for i, qa in enumerate(paper.get("qas", [])):
            question = qa.get("question", "")
            # Get first answerer's free-form answer
            answers = qa.get("answers", [])
            answer_text = ""
            for ans in answers:
                a = ans.get("answer", {})
                if a.get("free_form_answer"):
                    answer_text = a["free_form_answer"]
                    break
                if a.get("extractive_spans"):
                    answer_text = " ".join(a["extractive_spans"])
                    break
            qas.append(QAPair(
                q_id=f"qasper_{paper_id}_{i}",
                question=question,
                answer=answer_text,
            ))

        if full_text and qas:
            documents.append(Document(
                doc_id=paper_id,
                text=full_text,
                questions=qas,
                metadata={"title": paper.get("title", "")},
            ))
            if max_docs and len(documents) >= max_docs:
                break

    logger.info("Loaded %d QASPER papers with %d QA pairs",
                len(documents), sum(len(d.questions) for d in documents))
    return documents


# ============================================================
# CUAD — direct download from Atticus Project
# ============================================================

_CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def load_cuad(max_docs: int | None = None) -> list[Document]:
    """Load CUAD from direct download."""
    raw = data_dir("raw/cuad")
    zip_path = _download_file(_CUAD_URL, raw / "data.zip", "CUAD contracts")

    # Extract if not already done
    extracted = raw / "extracted"
    if not extracted.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)

    # Find the JSON file (SQuAD format)
    json_files = list(extracted.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON files found in CUAD archive")

    # Use the first/largest JSON (train set)
    json_path = max(json_files, key=lambda p: p.stat().st_size)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for article in data.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            qas = []
            for qa in para.get("qas", []):
                answers = qa.get("answers", [])
                answer_text = answers[0]["text"] if answers else ""
                qas.append(QAPair(
                    q_id=qa.get("id", ""),
                    question=qa.get("question", ""),
                    answer=answer_text,
                    metadata={"is_impossible": qa.get("is_impossible", False)},
                ))
            if context and qas:
                doc_id = f"cuad_{title}_{len(documents)}"
                documents.append(Document(
                    doc_id=doc_id,
                    text=context,
                    questions=qas,
                    metadata={"title": title},
                ))
                if max_docs and len(documents) >= max_docs:
                    break
        if max_docs and len(documents) >= max_docs:
            break

    logger.info("Loaded %d CUAD contract sections with %d QA pairs",
                len(documents), sum(len(d.questions) for d in documents))
    return documents


# ============================================================
# QuALITY — direct download from NYU
# ============================================================

_QUALITY_BASE = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped"


def load_quality(split: str = "dev", max_docs: int | None = None) -> list[Document]:
    """Load QuALITY from direct download."""
    raw = data_dir("raw/quality")
    url = f"{_QUALITY_BASE}.{split}"
    path = _download_file(url, raw / f"quality_{split}.jsonl", f"QuALITY {split}")

    doc_map: dict[str, Document] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            article_id = entry.get("article_id", "")
            if article_id not in doc_map:
                doc_map[article_id] = Document(
                    doc_id=f"quality_{article_id}",
                    text=entry.get("article", ""),
                    metadata={"title": entry.get("title", ""), "source": entry.get("source", "")},
                )
            doc = doc_map[article_id]
            for q in entry.get("questions", []):
                options = q.get("options", [])
                gold_idx = q.get("gold_label", 1) - 1
                answer = options[gold_idx] if 0 <= gold_idx < len(options) else ""
                doc.questions.append(QAPair(
                    q_id=f"quality_{article_id}_{q.get('question_unique_id', '')}",
                    question=q.get("question", ""),
                    answer=answer,
                    metadata={"options": options, "difficulty": q.get("difficult", 0)},
                ))
            if max_docs and len(doc_map) >= max_docs:
                break

    documents = list(doc_map.values())
    logger.info("Loaded %d QuALITY articles with %d QA pairs",
                len(documents), sum(len(d.questions) for d in documents))
    return documents


# ============================================================
# Unified loader
# ============================================================

_SCATTER_QUESTION_TEMPLATES = {
    "distributed_attributes": "Provide a comprehensive description of {entity} including their appearance, background, personality, relationships, and development throughout the story.",
    "progressive_accumulation": "How does {entity} change or develop over the course of the narrative?",
    "cross_reference": "Describe all significant relationships that {entity} has with other characters and how these relationships evolve.",
    "implicit": "What motivates {entity}? Explain their goals, desires, and the reasoning behind their key decisions.",
    "contradictory_evolution": "Are there any contradictions or inconsistencies in how {entity} is portrayed throughout the text? If so, describe them.",
}


def load_scatterqa(max_docs: int | None = None, **kwargs) -> list[Document]:
    """Load ScatterQA benchmark from gold evidence JSONL.

    Groups records by novel (Gutenberg book ID) and loads full novel text.
    """
    from src.data.gutenberg import download_gutenberg_text
    from src.evaluation.gold_schema import load_gold_evidence

    cleaned_path = data_dir() / "scatterqa" / "gold_evidence_cleaned.jsonl"
    gold_path = cleaned_path if cleaned_path.exists() else data_dir() / "scatterqa" / "gold_evidence.jsonl"
    if not gold_path.exists():
        logger.warning("ScatterQA data not found at %s — dataset still building?", gold_path)
        return []

    records = load_gold_evidence(gold_path)
    if not records:
        logger.warning("ScatterQA gold_evidence.jsonl is empty")
        return []

    # Group records by novel (gutenberg_{book_id})
    novel_records: dict[str, list] = {}
    for rec in records:
        # entity_id format: "gutenberg_{book_id}:{entity_name}"
        novel_key = rec.entity_id.split(":")[0]  # "gutenberg_1342"
        novel_records.setdefault(novel_key, []).append(rec)

    documents = []
    for novel_key, recs in novel_records.items():
        # Extract book_id for Gutenberg download
        book_id_str = novel_key.replace("gutenberg_", "")
        try:
            book_id = int(book_id_str)
        except ValueError:
            logger.warning("Cannot parse book_id from %s, skipping", novel_key)
            continue

        # Download full novel text (cached)
        try:
            text = download_gutenberg_text(book_id)
        except Exception as e:
            logger.warning("Failed to download %s: %s", novel_key, e)
            continue

        # Build QA pairs with proper question text
        qas = []
        for rec in recs:
            entity = rec.query_metadata.gold_entity
            cat = rec.query_metadata.scatter_category
            question = _SCATTER_QUESTION_TEMPLATES.get(cat, "Describe {entity} in detail.").format(entity=entity)

            answer_parts = [f"{attr}: {evidence}" for attr, evidence in rec.gold_attributes.items()]
            answer = " | ".join(answer_parts) if answer_parts else ""

            qas.append(QAPair(
                q_id=rec.query_id,
                question=question,
                answer=answer,
                metadata={
                    "entity_id": rec.entity_id,
                    "gold_chunk_ids": rec.gold_chunk_ids,
                    "scatter_category": cat,
                    "gold_entity": entity,
                },
            ))

        documents.append(Document(
            doc_id=novel_key,
            text=text,
            questions=qas,
            metadata={"source": "scatterqa", "domain": "benchmark"},
        ))
        if max_docs and len(documents) >= max_docs:
            break

    logger.info("Loaded %d ScatterQA novels with %d QA pairs",
                len(documents), sum(len(d.questions) for d in documents))
    return documents


LOADERS = {
    "narrativeqa": load_narrativeqa,
    "qasper": load_qasper,
    "cuad": load_cuad,
    "quality": load_quality,
    "scatterqa": load_scatterqa,
}


def load_dataset(name: str, **kwargs) -> list[Document]:
    """Load any dataset by name. Deduplicates by doc_id."""
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADERS.keys())}")
    docs = LOADERS[name](**kwargs)
    seen = set()
    deduped = []
    for doc in docs:
        if doc.doc_id not in seen:
            seen.add(doc.doc_id)
            deduped.append(doc)
    if len(deduped) < len(docs):
        logger.info("Deduplication removed %d/%d documents for %s",
                     len(docs) - len(deduped), len(docs), name)
    return deduped

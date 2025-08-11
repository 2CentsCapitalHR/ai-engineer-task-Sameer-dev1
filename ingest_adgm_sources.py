# ingestion_adgm_source.py - Updated to store summarized chunks
import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

import numpy as np
try:
    import faiss
except Exception as e:
    raise ImportError("Please install faiss-cpu: pip install faiss-cpu") from e

import requests

logger = logging.getLogger("ingestion_adgm_source")
logging.basicConfig(level=logging.INFO)

INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "faiss_index"))
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

RAW_TEXT_DIR = INDEX_DIR / "raw_texts"
PROCESSED_META_FILE = INDEX_DIR / "processed_sources.json"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbedderREST:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        self.model = EMBEDDING_MODEL
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"

    def embed(self, texts: List[str]) -> np.ndarray:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        vectors = []
        for text in texts:
            body = {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]}
            }
            for attempt in range(3):
                try:
                    resp = requests.post(self.url, headers=headers, json=body, timeout=60)
                    if resp.status_code != 200:
                        raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text}")
                    data = resp.json()
                    if "embedding" in data and "values" in data["embedding"]:
                        embedding_vector = data["embedding"]["values"]
                        vectors.append([float(x) for x in embedding_vector])
                        break
                    else:
                        raise RuntimeError("Unexpected embedding response format: " + str(data.keys()))
                except Exception as e:
                    logger.warning(f"Embedding API call failed attempt {attempt+1}/3: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            else:
                raise RuntimeError("Failed to get embeddings after retries")
        return np.array(vectors, dtype=np.float32)


class VectorStore:
    def __init__(self, index_dir: Path = INDEX_DIR, dim: int = EMBEDDING_DIM):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.faiss"
        self.meta_path = self.dir / "metadata.json"
        self.dim = dim
        self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists():
            logger.info("Loading existing FAISS index.")
            self.index = faiss.read_index(str(self.index_path))
            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = []
        else:
            logger.info("Creating new FAISS index (IndexFlatIP).")
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metas)
        self.save()


class Ingestor:
    def __init__(self, embedder: EmbedderREST, store: VectorStore, chunk_size_tokens: int = 500, overlap_tokens: int = 50):
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size_tokens
        self.overlap = overlap_tokens

    def _simple_tokenizer(self, text: str) -> List[str]:
        return text.split()

    def _detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def _chunk_text(self, text: str) -> List[str]:
        tokens = self._simple_tokenizer(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._detokenize(chunk_tokens))
            start += (self.chunk_size - self.overlap)
        return chunks

    def _summarize_chunk(self, chunk: str) -> str:
        """Simple local summary: take first 2 sentences or 300 chars."""
        sentences = chunk.split(". ")
        summary = ". ".join(sentences[:2])
        if len(summary) > 300:
            summary = summary[:300] + "..."
        return summary

    def ingest_text(self, title: str, text: str, metadata: Dict[str, Any]):
        doc_id = sha1(title + str(time.time()))
        chunks = self._chunk_text(text)
        metas = []
        for i, ch in enumerate(chunks):
            short_preview = self._summarize_chunk(ch)  # store summarized version
            m = metadata.copy()
            m.update({
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_text": ch,
                "chunk_preview": short_preview  # replaced with summary
            })
            metas.append(m)
        BATCH_SIZE = 10
        all_embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            embeddings = self.embedder.embed(batch)
            all_embeddings.extend(embeddings)
        vectors = np.array(all_embeddings, dtype=np.float32)
        self.store.add(vectors, metas)
        logger.info(f"Ingested {len(chunks)} chunks for {title} (summarized stored)")
        return doc_id, len(chunks)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True, help="Path to JSON with source data")
    args = parser.parse_args()

    embedder = EmbedderREST()
    store = VectorStore()
    ingestor = Ingestor(embedder, store)

    with open(args.source_json, "r", encoding="utf-8") as f:
        sources = json.load(f)

# ✅ Auto-convert list format to dict format
    if isinstance(sources, list):
        try:
            sources = {entry["url"]: entry["text"] for entry in sources if "url" in entry and "text" in entry}
            logger.info(f"Converted source list to dictionary format with {len(sources)} entries.")
        except Exception as e:
            raise ValueError(f"Invalid JSON list format: {e}")
        except Exception as e:
            raise ValueError(f"Invalid JSON list format: {e}")
    elif not isinstance(sources, dict):
        raise ValueError("Invalid JSON format: must be a dict or list of {url, text} objects")


    for url, text in sources.items():
        ingestor.ingest_text(url, text, {"source_url": url, "source_filename": Path(url).name})

    logger.info("Ingestion complete with summaries.")

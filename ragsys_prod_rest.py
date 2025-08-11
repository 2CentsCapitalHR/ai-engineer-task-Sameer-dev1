# ragsys_prod_rest.py - Optimized for short inline comments with summarization
import os
import json
import time
import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

import numpy as np
try:
    import faiss
except Exception as e:
    raise ImportError("Please install faiss-cpu: pip install faiss-cpu") from e

import requests
from PyPDF2 import PdfReader

logger = logging.getLogger("ragsys_prod_rest")
logging.basicConfig(level=logging.INFO)

INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "faiss_index"))
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
# Use more stable model - gemini-1.5-flash instead of 2.0
GENERATOR_MODEL = os.getenv("GEMINI_GENERATOR_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))
TOP_K = int(os.getenv("RAG_TOP_K", "6"))

RAW_TEXT_DIR = INDEX_DIR / "raw_texts"
PROCESSED_META_FILE = INDEX_DIR / "processed_sources.json"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbedderREST:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        self.model = EMBEDDING_MODEL
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        self.dim = EMBEDDING_DIM
        logger.info(f"Initialized EmbedderREST with model: {self.model}, dim: {self.dim}")
        logger.info(f"Embedding URL: {self.url}")

    def embed(self, texts: List[str]) -> np.ndarray:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        vectors = []
        for text in texts:
            # Ensure text is not empty and not too long
            if not text or len(text) < 1:
                logger.warning("Empty text provided for embedding, using fallback")
                fallback_vector = [0.0] * self.dim
                vectors.append(fallback_vector)
                continue
                
            # Truncate text if too long (Gemini has limits)
            if len(text) > 60000:  # Gemini has ~60k character limit
                logger.warning(f"Text too long ({len(text)} chars), truncating to 60000 chars")
                text = text[:60000]
                
            body = {
                "model": self.model,
                "content": {"parts": [{"text": text}]}
            }
            logger.info(f"Embedding request body model parameter: {body['model']}")
            for attempt in range(3):
                try:
                    resp = requests.post(self.url, headers=headers, json=body, timeout=60)
                    if resp.status_code == 429:  # Rate limit
                        wait_time = 2 ** attempt * 10  # 10, 20, 40 seconds
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    elif resp.status_code != 200:
                        logger.error(f"Embedding API error {resp.status_code}: {resp.text}")
                        if attempt < 2:
                            time.sleep(2 ** attempt * 5)  # 5, 10 seconds
                            continue
                        else:
                            # Use fallback on final attempt
                            logger.error("Failed to get embeddings after retries, using fallback")
                            fallback_vector = [0.0] * self.dim
                            vectors.append(fallback_vector)
                            break
                    
                    data = resp.json()
                    if "embedding" in data and "values" in data["embedding"]:
                        embedding_vector = data["embedding"]["values"]
                        vectors.append([float(x) for x in embedding_vector])
                        # Add small delay between successful requests
                        time.sleep(1)
                        break
                    else:
                        logger.error("Unexpected embedding response format: " + str(data.keys()))
                        if attempt < 2:
                            time.sleep(2 ** attempt * 5)
                            continue
                except Exception as e:
                    logger.warning(f"Embedding API call failed attempt {attempt+1}/3: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt * 5)  # 5, 10 seconds
                    else:
                        # If all retries failed, return a fallback
                        logger.error("Failed to get embeddings after retries, using fallback")
                        fallback_vector = [0.0] * self.dim
                        vectors.append(fallback_vector)
            else:
                # If we got here without breaking, all attempts failed
                logger.error("Failed to get embeddings after all retries, using fallback")
                fallback_vector = [0.0] * self.dim
                vectors.append(fallback_vector)
        return np.array(vectors, dtype=np.float32)


class VectorStore:
    def __init__(self, index_dir: Path = INDEX_DIR, dim: int = EMBEDDING_DIM):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.faiss"
        self.meta_path = self.dir / "metadata.json"
        self.processed_meta_path = PROCESSED_META_FILE
        self.dim = dim
        self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists():
            logger.info("Loading existing FAISS index from ingestion script.")
            self.index = faiss.read_index(str(self.index_path))
            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = self._reconstruct_metadata()
        else:
            logger.info("Creating new FAISS index (IndexFlatIP).")
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def _reconstruct_metadata(self) -> List[Dict]:
        metadata = []
        if RAW_TEXT_DIR.exists():
            for text_file in RAW_TEXT_DIR.glob("*.txt"):
                try:
                    with open(text_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    source_url = "unknown"
                    if self.processed_meta_path.exists():
                        with open(self.processed_meta_path, "r") as f:
                            processed = json.load(f)
                            filename_hash = text_file.stem
                            for url in processed.keys():
                                if hashlib.sha256(url.encode()).hexdigest().startswith(filename_hash[:8]):
                                    source_url = url
                                    break
                    metadata.append({
                        "source_url": source_url,
                        "source_filename": text_file.name,
                        "chunk_preview": content[:300],
                        "doc_id": text_file.stem,
                        "chunk_index": 0
                    })
                except Exception as e:
                    logger.warning(f"Could not process {text_file}: {e}")
        logger.info(f"Reconstructed {len(metadata)} metadata entries")
        return metadata

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metas)
        self.save()

    def search(self, vector: np.ndarray, top_k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Run ingestion script first.")
            return []
        v = vector.astype(np.float32)
        faiss.normalize_L2(v)
        D, I = self.index.search(v, top_k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append((self.metadata[idx], float(score)))
        return results


class RAGManager:
    def __init__(self, embedder: EmbedderREST, store: VectorStore, generator_model: str = GENERATOR_MODEL):
        self.embedder = embedder
        self.store = store
        self.generator_model = generator_model
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

    def test_basic_generation(self):
        """Test method to verify basic LLM functionality"""
        simple_prompt = "Please respond with a simple JSON object containing a greeting. Format: {\"message\": \"hello\"}"
        response = self.generate(simple_prompt, max_tokens=2048)
        logger.info(f"Basic test result: {response}")
        return response

    def summarize_chunk(self, text: str) -> str:
        """Lightweight local summarization (no extra API calls)"""
        sentences = text.split(". ")
        summary = ". ".join(sentences[:2])  # Take first 2 sentences
        if len(summary) > 300:
            summary = summary[:300] + "..."
        return summary

    def retrieve(self, query: str, top_k: int = 1):
        try:
            qvec = self.embedder.embed([query])
            if np.all(qvec == 0):
                logger.error("Embedding failed: returned zero vector. Check Gemini API key and quota.")
                return [{
                    "chunk": "",
                    "meta": {"error": "Embedding failed: zero vector returned. Check Gemini API key and quota."},
                    "score": 0
                }]
            hits = self.store.search(qvec, top_k)
            results = []
            for hit_meta, score in hits:
                chunk_content = hit_meta.get("chunk_text", hit_meta.get("chunk_preview", ""))
                results.append({
                    "chunk": chunk_content,
                    "meta": hit_meta,
                    "score": score
                })
            if not results:
                logger.error("FAISS retrieval returned no results. Index may be empty or embedding mismatch.")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return [{
                "chunk": "",
                "meta": {"error": f"Retrieval failed: {e}"},
                "score": 0
            }]

    def build_prompt(self, doc_text: str, doc_type: str, retrieved: List[Dict[str, Any]]):
        # Simplified, less triggering prompt
        snippet = doc_text[:500] if len(doc_text) > 500 else doc_text
        
        parts = [
            "You are a helpful document analysis assistant.",
            "Review the following document and provide structured feedback.",
            f"Document Type: {doc_type}",
            f"Document Text: {snippet}"
        ]
        
        # Add context if available but keep it simple
        if retrieved and len(retrieved) > 0:
            context_parts = []
            for i, source in enumerate(retrieved[:1]):  # Use only 1 source to reduce complexity
                if 'chunk' in source and source['chunk']:
                    context_chunk = source['chunk'][:200] + "..." if len(source['chunk']) > 200 else source['chunk']
                    context_parts.append(f"Reference: {context_chunk}")
            
            if context_parts:
                parts.append("Related information:")
                parts.extend(context_parts)
        
        # Simpler JSON format request
        parts.append("""Please provide your analysis in this JSON format:
{
  "summary": "Brief summary of the document",
  "comments": [
    {
      "paragraph_id": 1,
      "comment": "Your observation here",
      "severity": "low",
      "suggestion": "Your suggestion here"
    }
  ]
}""")
        
        parts.append("Return only valid JSON, no other text.")
        
        return "\n\n".join(parts)

    def generate(self, prompt: str, max_tokens: int = 2048):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.generator_model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        logger.info(f"Generation URL: {url}")
        logger.info(f"Using generator model: {self.generator_model}")
        
        # Ensure prompt is not too long
        if len(prompt) > 25000:  # More conservative limit
            logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to 25000 chars")
            prompt = prompt[:25000]
            
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,  # Now defaults to 2048
                "temperature": 0.2,  # Slightly higher for better generation
                "topP": 0.9,  # Increased for better diversity
                "topK": 40
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        # Add retries for API calls with better debugging
        for attempt in range(4):  # Increased to 4 attempts as recommended
            try:
                logger.info(f"Attempting generation (attempt {attempt+1}/4) with model: {self.generator_model}")
                resp = requests.post(url, headers=headers, json=body, timeout=90)  # Increased timeout
                
                # Debug logging
                logger.info(f"Response status: {resp.status_code}")
                
                if resp.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt * 10  # 10, 20, 40, 80 seconds
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif resp.status_code == 400:
                    logger.error(f"Bad request error: {resp.text}")
                    # Try with shorter prompt
                    if len(prompt) > 10000:
                        prompt = prompt[:10000] + "..."
                        body["contents"] = [{"parts": [{"text": prompt}]}]
                        continue
                    else:
                        return json.dumps({"error": f"Bad request: {resp.text}"})
                elif resp.status_code != 200:
                    logger.error(f"Gemini generation API error {resp.status_code}: {resp.text}")
                    if attempt < 3:
                        time.sleep(2 ** attempt * 5)  # 5, 10, 20 seconds
                        continue
                    return json.dumps({"error": f"API error {resp.status_code}: {resp.text}"})
                
                data = resp.json()
                logger.info(f"Response data keys: {list(data.keys())}")
                
                candidates = data.get("candidates", [])
                if not candidates:
                    logger.error(f"No candidates in Gemini response: {data}")
                    if attempt < 3:
                        time.sleep(2 ** attempt * 5)
                        continue
                    return json.dumps({"error": "No candidates in response", "raw_response": str(data)[:300]})
                
                candidate = candidates[0]
                finish_reason = candidate.get("finishReason", "")
                logger.info(f"Finish reason: {finish_reason}")
                
                # Handle various finish reasons
                if finish_reason in ["SAFETY", "RECITATION"]:
                    logger.warning(f"Content blocked by safety filters: {finish_reason}")
                    # Try with even simpler prompt
                    if attempt < 3:
                        simple_prompt = f"Analyze this text and provide JSON feedback: {prompt[:200] if len(prompt) > 200 else prompt}"
                        body["contents"] = [{"parts": [{"text": simple_prompt}]}]
                        time.sleep(5)
                        continue
                    return json.dumps({
                        "error": f"Content blocked: {finish_reason}",
                        "comments": [{"paragraph_id": 0, "comment": "Content blocked by safety filters", "severity": "medium", "suggestion": "Try with different content"}]
                    })
                
                if finish_reason == "MAX_TOKENS":
                    logger.warning("Response truncated due to max tokens - increasing limit")
                    if max_tokens < 4096:
                        max_tokens = 4096
                        body["generationConfig"]["maxOutputTokens"] = max_tokens
                        if attempt < 3:
                            continue
                
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                logger.info(f"Content structure: role={content.get('role')}, parts_count={len(parts) if parts else 0}")
                
                if parts and isinstance(parts, list) and len(parts) > 0:
                    text_content = parts[0].get("text", "").strip()
                    
                    logger.info(f"Generated text length: {len(text_content)}")
                    logger.info(f"Text preview: {text_content[:200]}...")
                    
                    # Handle empty response
                    if not text_content:
                        logger.warning("Gemini returned empty text content")
                        if attempt < 3:
                            # Try with even simpler prompt
                            simple_prompt = "Please respond with JSON: {\"message\": \"test response\"}"
                            body["contents"] = [{"parts": [{"text": simple_prompt}]}]
                            time.sleep(5)
                            continue
                        return json.dumps({
                            "error": "Empty response from Gemini",
                            "comments": [{"paragraph_id": 0, "comment": "AI returned empty response", "severity": "medium", "suggestion": "Try with different content"}]
                        })
                    
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(text_content)
                        logger.info("Successfully parsed JSON response")
                        return text_content
                    except json.JSONDecodeError:
                        logger.warning("Response is not valid JSON, trying to extract...")
                        # Try to extract JSON from the text
                        json_match = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', text_content, re.DOTALL)
                        if json_match:
                            try:
                                json_str = json_match.group(0)
                                parsed = json.loads(json_str)
                                logger.info("Successfully extracted JSON from response")
                                return json_str
                            except:
                                logger.warning("Extracted text is still not valid JSON")
                        
                        # If we can't parse JSON but got text, return it with error flag
                        return json.dumps({
                            "error": "Invalid JSON format",
                            "raw_text": text_content[:500] + "..." if len(text_content) > 500 else text_content,
                            "comments": [{"paragraph_id": 0, "comment": "Response format issue", "severity": "low", "suggestion": "Manual review needed"}]
                        })
                
                # Handle empty parts or missing content
                logger.error(f"No usable content in response. Content: {content}")
                if attempt < 3:
                    time.sleep(2 ** attempt * 5)
                    continue
                
                return json.dumps({
                    "error": "No usable content in response",
                    "comments": [{"paragraph_id": 0, "comment": "Failed to generate analysis", "severity": "high", "suggestion": "Try again or contact support"}]
                })
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout (attempt {attempt+1}/4)")
                if attempt < 3:
                    time.sleep(2 ** attempt * 10)
                    continue
                return json.dumps({"error": "Request timeout", "comments": []})
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt+1}/4): {str(e)}")
                if attempt < 3:
                    time.sleep(2 ** attempt * 5)
                else:
                    return json.dumps({
                        "error": f"Request failed: {str(e)}",
                        "comments": [{"paragraph_id": 0, "comment": "Analysis failed", "severity": "high", "suggestion": "Contact support"}]
                    })
        
        # If all attempts failed
        return json.dumps({
            "error": "All API attempts failed",
            "comments": [{"paragraph_id": 0, "comment": "Service unavailable", "severity": "high", "suggestion": "Try again later"}]
        })

    def analyze_document(self, doc_text: str, doc_type: str, top_k: int = 1):
        # Handle empty or None input
        if not doc_text:
            return {
                "error": "Empty document text",
                "comments": [{"paragraph_id": 0, "comment": "No text to analyze", "severity": "high", "suggestion": "Please provide document text"}]
            }
        
        # Test basic generation first
        logger.info("Testing basic LLM generation...")
        test_result = self.test_basic_generation()
        try:
            json.loads(test_result)
            logger.info("Basic generation test passed")
        except:
            logger.warning("Basic generation test failed, but continuing with document analysis")
            
        # Truncate input for better processing
        query_text = doc_text[:800] if len(doc_text) > 800 else doc_text
        
        # Try to retrieve relevant context
        try:
            retrieved = self.retrieve(query_text, top_k=top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            retrieved = []
            
        # Build prompt with available context
        prompt = self.build_prompt(query_text, doc_type, retrieved)
        
        # Generate response with higher token limit
        gen = self.generate(prompt, max_tokens=2048)
        
        # Try to parse the response as JSON
        try:
            result = json.loads(gen)
            logger.info("Successfully parsed document analysis result")
            return result
        except json.JSONDecodeError:
            logger.error("Failed to parse final result as JSON")
            # Return a structured fallback
            return {
                "error": "JSON parsing failed",
                "raw_response": gen[:500] if isinstance(gen, str) else str(gen)[:500],
                "summary": "Analysis completed but format parsing failed",
                "comments": [{"paragraph_id": 0, "comment": "Could not parse AI response", "severity": "medium", "suggestion": "Check response format"}]
            }


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def check_knowledge_base(index_dir: Path = INDEX_DIR) -> bool:
    return (index_dir / "index.faiss").exists()


def get_knowledge_base_stats(index_dir: Path = INDEX_DIR) -> Dict[str, Any]:
    stats = {"exists": False, "total_vectors": 0, "processed_sources": 0}
    index_path = index_dir / "index.faiss"
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            stats["exists"] = True
            stats["total_vectors"] = index.ntotal
        except Exception as e:
            logger.error(f"Error reading index: {e}")
    if PROCESSED_META_FILE.exists():
        try:
            with open(PROCESSED_META_FILE, "r") as f:
                processed = json.load(f)
                stats["processed_sources"] = len(processed)
        except Exception as e:
            logger.error(f"Error reading processed sources: {e}")
    return stats


if __name__ == "__main__":
    # Test basic functionality first
    print("Testing Gemini API connection...")
    
    try:
        embedder = EmbedderREST()
        store = VectorStore()
        rag_manager = RAGManager(embedder, store)
        
        # Test basic generation
        print("Testing basic LLM generation...")
        test_result = rag_manager.test_basic_generation()
        print(f"Basic test result: {test_result}")
        
        if not check_knowledge_base():
            print("No FAISS index found. Please run:")
            print("python ingestion_adgm_source.py --source-json adgm_sources.json")
        else:
            stats = get_knowledge_base_stats()
            print(f"Knowledge Base Stats:\n  • Total vectors: {stats['total_vectors']}\n  • Processed sources: {stats['processed_sources']}")
        
        # Test document analysis
        sample_doc = "This is a sample business document for testing analysis capabilities."
        print("Testing document analysis...")
        result = rag_manager.analyze_document(sample_doc, "sample_doc")
        print(f"Analysis Result: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        logger.exception("Full error details:")
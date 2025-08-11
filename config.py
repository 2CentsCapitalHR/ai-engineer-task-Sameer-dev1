"""
Configuration file for ADGM Corporate Agent RAG system (Gemini only)
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the ADGM Corporate Agent (Gemini only)"""
    
    # LLM Provider (fixed to Gemini)
    LLM_PROVIDER = "gemini"
    
    # API Key (set as environment variable)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Model Configuration
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
    
    # RAG Configuration
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    TOP_K_REFERENCES = int(os.getenv("TOP_K_REFERENCES", "3"))
    
    # Cache Configuration
    CACHE_DIR = os.getenv("CACHE_DIR", "adgm_cache")
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    # Document Processing
    MAX_PREVIEW_CHARS = int(os.getenv("MAX_PREVIEW_CHARS", "1200"))
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration (Gemini only)"""
        return {
            "provider": cls.LLM_PROVIDER,
            "api_key": cls.GEMINI_API_KEY,
            "model": cls.GEMINI_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "max_context_length": cls.MAX_CONTEXT_LENGTH,
            "top_k_references": cls.TOP_K_REFERENCES
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.GEMINI_API_KEY:
            print("  Warning: GEMINI_API_KEY not set. Gemini features will be limited.")
            return False
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 ADGM Corporate Agent Configuration (Gemini only):")
        print(f"  LLM Provider: {cls.LLM_PROVIDER}")
        print(f"  Gemini Model: {cls.GEMINI_MODEL}")
        print(f"  Max Context Length: {cls.MAX_CONTEXT_LENGTH}")
        print(f"  Max Tokens: {cls.MAX_TOKENS}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print(f"  Top K References: {cls.TOP_K_REFERENCES}")
        print(f"  Cache Directory: {cls.CACHE_DIR}")
        print(f"  Enable Caching: {cls.ENABLE_CACHING}")
        print(f"  Max Preview Chars: {cls.MAX_PREVIEW_CHARS}")
        print(f"  Max Chunk Size: {cls.MAX_CHUNK_SIZE}")

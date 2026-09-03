import os

# Azure AI Search configuration
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT", "")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "rag-index")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY", "")

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# MCP Server configuration
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:9001")

# TC1 -- Retrieval Recall
# BUG: minimum Azure Search score set unrealistically high -- filters out ALL real matches
MIN_SEARCH_SCORE = float(os.environ.get("MIN_SEARCH_SCORE", "10.0"))

# TC2 -- Distraction / near-duplicate filtering
# BUG: Jaccard similarity threshold set too high -- near-duplicate documents are never
# recognised as duplicates of each other, so both get cited
DEDUP_JACCARD_THRESHOLD = float(os.environ.get("DEDUP_JACCARD_THRESHOLD", "0.99"))

# TC5 -- Budget & Overflow
# BUG: context character budget set far too small -- truncates the LLM's context to almost nothing
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "80"))

# TC6/TC7 -- Conversation memory (accumulation, staleness)
# BUG: history cap set unrealistically high -- old turns are never evicted within a
# normal test session, so memory grows without bound
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "10000"))

# TC12/TC18 -- Grounding & Attribution / User Prompt Verification
# BUG: citation markers disabled -- the LLM can't reference [1], [2] etc. in its answer
INCLUDE_CITATION_MARKERS = False

# TC14 -- Query Rewriting / Expansion
# BUG: threshold set too low -- a genuinely short/ambiguous query (e.g. 2 words) is
# still treated as "long enough" and never gets expanded before retrieval
SHORT_QUERY_WORD_THRESHOLD = int(os.environ.get("SHORT_QUERY_WORD_THRESHOLD", "1"))

# TC12/TC18 -- Grounding & Attribution / User Prompt Verification
# BUG: system prompt explicitly INVITES the model to use its own general knowledge --
# defeats the whole point of RAG grounding. It is also missing the numbered-citation
# reference instruction entirely.
DEFAULT_SYSTEM_PROMPT = """You are an enterprise AI assistant with dual routing capabilities:


1. GENERAL KNOWLEDGE / FACTUAL / TECHNICAL QUESTIONS (Programming languages like "What is Python?", algorithms, math, science, general definitions):
   - If the question is purely general knowledge and unrelated to internal organizational policy, you ARE permitted to use your own knowledge base to provide a clear, concise, and helpful answer, even if the retrieved context is empty or unrelated.
Retrieved context:
$search_results$

Question: $query$
Answer:"""

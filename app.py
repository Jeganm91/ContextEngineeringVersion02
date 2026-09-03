import os
import re
import json
import time
import logging
import requests
from collections import defaultdict
from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import config

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_observability")

# TC10 -- MCP Routing (Tool Surface)
# BUG: these trigger words don't match how users actually ask for current information
MCP_TRIGGERS = ["unwanted", "old version", "last updation", "old", "expired"]

# --- Safety filters for retrieved documents ---
# TC3 -- Document Exclusion Filtering
# BUG: both the PII-filename marker and the DEPRECATED-content marker have been
# removed from these lists, so documents that should never be cited slip through
EXCLUDED_FILENAME_MARKERS = []
EXCLUDED_CONTENT_MARKERS = [
    "internal-confidential",
    "should never be surfaced",
    "must not be included, quoted, or paraphrased",
]

# TC9 -- Injection Resistance
# BUG: one important pattern is missing from this list
_INJECTION_PATTERNS = [
    r"disregard (all|any|the) (previous|prior|above) (instructions|prompt)",
    r"reveal (the|your) (system prompt|instructions)",
    r"you are now",
    r"forget (everything|all) (you|that)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# TC4 -- Authority Conflict / Contradiction Detection
# BUG: the pair that would catch the real contradiction in this lab's knowledge
# base (premium license renewal every "12 months" per one document vs "6 months"
# per another, both marked CURRENT) is missing from this list
_CONTRADICTION_PAIRS = [
    ("deprecated", "active"),
    ("discontinued", "in effect"),
]

# TC6/TC7/TC8/TC17 -- Conversation memory & subject scoping
# Each session entry stores its owning user_id alongside its turns.
_SESSIONS = {}


class ConfigurationError(Exception):
    """Raised when a required Azure Search / Azure OpenAI setting is missing."""
    pass


def is_document_excluded(content: str, filename: str) -> bool:
    filename_lower = (filename or "").lower()
    content_lower = (content or "").lower()

    for marker in EXCLUDED_FILENAME_MARKERS:
        if marker.lower() in filename_lower:
            return True

    for marker in EXCLUDED_CONTENT_MARKERS:
        if marker.lower() in content_lower:
            return True

    return False


def should_use_mcp(query: str) -> bool:
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in MCP_TRIGGERS)


def call_mcp_tool(query: str):
    try:
        url = f"{config.MCP_SERVER_URL}/tools/get_latest_pricing"
        response = requests.post(url, json={"query": query}, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"MCP Invocation Error: {e}")
    return None


def scan_for_injection(text: str) -> bool:
    if not text:
        return False
    return bool(_INJECTION_RE.search(text))


def detect_contradictions(citations: list) -> list:
    found = []
    for i, c1 in enumerate(citations):
        for c2 in citations[i + 1:]:
            c1l, c2l = c1["content"].lower(), c2["content"].lower()
            for term_a, term_b in _CONTRADICTION_PAIRS:
                if (term_a in c1l and term_b in c2l) or (term_b in c1l and term_a in c2l):
                    found.append({
                        "source_a": c1["source"],
                        "source_b": c2["source"],
                        "signal": f"{term_a!r} vs {term_b!r}",
                    })
    return found


def _jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate_citations(candidates: list) -> list:
    # TC2 -- Distraction: drop near-duplicate chunks so the same information
    # doesn't occupy multiple citation slots and crowd out different content.
    deduped = []
    for cand in candidates:
        is_dup = any(
            _jaccard_similarity(cand["content"], kept["content"]) >= config.DEDUP_JACCARD_THRESHOLD
            for kept in deduped
        )
        if not is_dup:
            deduped.append(cand)
    return deduped


def rerank_and_reorder(citations: list) -> list:
    """TC15 -- Context Assembly & Ordering: re-rank by score, then place the
    strongest chunks at the start AND end of the list (lost-in-the-middle
    mitigation), leaving weaker chunks in the middle where the model pays
    less attention."""
    ranked = sorted(citations, key=lambda c: c["score"], reverse=True)
    result = [None] * len(ranked)
    left, right = 0, len(ranked) - 1
    for i, c in enumerate(ranked):
        if i % 2 == 0:
            result[left] = c
            left += 1
        else:
            result[right] = c
            right -= 1
    return result


def expand_query_if_short(query: str) -> str:
    """TC14 -- Query Rewriting / Expansion: a short, under-specified query
    retrieves poorly on its own, so add domain context terms before search."""
    words = query.strip().split()
    if len(words) <= config.SHORT_QUERY_WORD_THRESHOLD:
        return query + " software license request policy"
    return query


def _session_key(session_id: str) -> str:
    # TC8 -- Session Isolation
    # BUG: should return session_id itself so each conversation is isolated,
    # but returns a constant instead -- every session reads and writes the
    # same shared history, so one user's conversation bleeds into another's
    return "shared-session"


def get_session_history(session_id: str, user_id: str) -> list:
    if not session_id:
        return []
    entry = _SESSIONS.get(_session_key(session_id))
    if not entry:
        return []
    # TC17 -- Personalization / Subject Scoping
    # BUG: returns the stored history without checking that the requesting
    # user_id matches the user_id that originally created this session --
    # anyone who learns or guesses a session_id can read that conversation
    return entry["turns"]


def append_to_session(session_id: str, user_id: str, query: str, answer: str) -> None:
    if not session_id:
        return
    key = _session_key(session_id)
    if key not in _SESSIONS:
        _SESSIONS[key] = {"owner_user_id": user_id, "turns": []}
    _SESSIONS[key]["turns"].append({"query": query, "answer": answer})

    # TC7 -- Memory Staleness
    # BUG: keeps the OLDEST turns and drops the NEWEST ones when trimming,
    # so a freshly-corrected fact gets evicted while a stale one lingers
    if len(_SESSIONS[key]["turns"]) > config.MAX_HISTORY_TURNS:
        _SESSIONS[key]["turns"] = _SESSIONS[key]["turns"][:config.MAX_HISTORY_TURNS]


def build_history_block(session_id: str, user_id: str) -> str:
    history = get_session_history(session_id, user_id)
    if not history:
        return ""
    lines = ["Conversation history (most recent last):"]
    for turn in history:
        lines.append(f"User: {turn['query']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines) + "\n\n"


def search_azure_knowledge_base(query: str, top_k: int = 5):
    endpoint = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT") or getattr(config, "AZURE_SEARCH_SERVICE_ENDPOINT", "")
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME") or getattr(config, "AZURE_SEARCH_INDEX_NAME", "rag-index")
    api_key = os.environ.get("AZURE_SEARCH_API_KEY") or getattr(config, "AZURE_SEARCH_API_KEY", "")

    if not endpoint or not api_key or endpoint.startswith("<") or api_key.startswith("<"):
        raise ConfigurationError("Azure AI Search endpoint or API key is not configured.")

    try:
        search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )

        # TC14 -- Query Rewriting / Expansion
        search_query = expand_query_if_short(query)

        results = search_client.search(search_text=search_query, top=top_k * 3)

        candidates = []
        for doc in results:
            content = (
                doc.get("chunk")
                or doc.get("content")
                or doc.get("text")
                or doc.get("merged_content")
                or ""
            )
            if not content:
                content = str({k: v for k, v in doc.items() if not k.startswith("@") and k != "text_vector"})

            source = (
                doc.get("title")
                or doc.get("filepath")
                or doc.get("metadata_storage_name")
                or "Azure Blob kb-docs"
            )
            filename = doc.get("metadata_storage_name") or doc.get("filepath") or ""

            if is_document_excluded(content, filename):
                print(f"Skipping excluded document: {source}")
                continue

            score = doc.get("@search.score", 0.0)

            # TC1 -- Retrieval Recall
            if score < config.MIN_SEARCH_SCORE:
                continue

            candidates.append({
                "content": content,
                "source": source,
                "score": score
            })

        # TC2 -- Distraction
        deduped = deduplicate_citations(candidates)
        top_candidates = deduped[:top_k]

        # TC15 -- Context Assembly & Ordering
        # BUG: rerank_and_reorder() is defined above but never called here --
        # citations are returned in raw Search order instead
        return top_candidates
    except ConfigurationError:
        raise
    except Exception as e:
        print(f"Azure Search Error: {e}")
        raise ConfigurationError(f"Azure AI Search request failed: {e}")


def generate_azure_openai_response(query: str, context_chunks: list, history_block: str = ""):
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or getattr(config, "AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or getattr(config, "AZURE_OPENAI_API_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or getattr(config, "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or getattr(config, "AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not endpoint or not api_key or endpoint.startswith("<") or api_key.startswith("<"):
        raise ConfigurationError("Azure OpenAI endpoint or API key is not configured.")

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )

        # TC12 -- Grounding & Attribution
        if config.INCLUDE_CITATION_MARKERS:
            labeled = [f"[{i+1}] Source: {c['source']}\n{c['content']}" for i, c in enumerate(context_chunks)]
        else:
            labeled = [c["content"] for c in context_chunks]

        # TC5 -- Budget & Overflow
        combined_context = ""
        for block in labeled:
            if len(combined_context) + len(block) > config.MAX_CONTEXT_CHARS:
                break
            combined_context += block + "\n\n"

        combined_context = history_block + combined_context

        prompt_template = getattr(config, "DEFAULT_SYSTEM_PROMPT", "Context:\n$search_results$\n\nQuestion: $query$\nAnswer:")
        prompt = prompt_template.replace("$search_results$", combined_context).replace("$query$", query)

        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except ConfigurationError:
        raise
    except Exception as e:
        print(f"Azure OpenAI Error: {e}")
        raise ConfigurationError(f"Azure OpenAI request failed: {e}")


def run_rag_query(query: str, session_id: str = None, user_id: str = None):
    start_time = time.time()

    result = {
        "query": query,
        "answer": "",
        "citations": [],
        "contradictions": [],
        "mcp_used": False,
        "mcp_output": None,
        "injection_flagged": False,
        "session_id": session_id,
    }

    if not query or not query.strip():
        result["answer"] = "Please enter a question."
        return result, 200

    if should_use_mcp(query):
        result["mcp_used"] = True
        mcp_res = call_mcp_tool(query)
        result["mcp_output"] = mcp_res
        # TC11 -- Untrusted Tool Output
        # BUG: the response is trusted as-is with no check that it actually
        # succeeded -- a malformed or error-shaped MCP response still gets
        # treated as a valid answer
        if mcp_res:
            result["answer"] = mcp_res.get("content", str(mcp_res))
            return result, 200

    try:
        citations = search_azure_knowledge_base(query)
    except ConfigurationError as e:
        result["error"] = str(e)
        result["answer"] = "Azure AI Search configuration is missing or invalid."
        return result, 404

    result["citations"] = citations
    result["contradictions"] = detect_contradictions(citations)

    result["injection_flagged"] = any(scan_for_injection(c["content"]) for c in citations)

    # TC13 -- Failure Handling
    # BUG: replies with a wrong permissive message (should refuse strictly
    # instead of inviting the model to answer from general knowledge)
    if not citations and not result["mcp_used"]:
        result["answer"] = "Information about the requested topic is not available, answer from your own knowledge."
        return result, 200

    history_block = build_history_block(session_id, user_id)

    try:
        generated_answer = generate_azure_openai_response(query, citations, history_block)
    except ConfigurationError as e:
        result["error"] = str(e)
        result["answer"] = "Azure OpenAI configuration is missing or invalid."
        return result, 404

    result["answer"] = generated_answer
    append_to_session(session_id, user_id, query, generated_answer)

    # TC16 -- Production Observability & Efficiency
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    result["latency_ms"] = elapsed_ms
    # BUG: latency is measured above, but never actually logged anywhere --
    # the structured observability record below is missing
    # logger.info(json.dumps({
    #     "latency_ms": elapsed_ms,
    #     "citation_count": len(result["citations"]),
    #     "mcp_used": result["mcp_used"],
    # }))

    return result, 200


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/query", methods=["POST", "GET"])
def query_api():
    if request.method == "GET":
        return render_template("index.html", result=None)

    data = request.get_json(silent=True)
    if not data:
        data = request.form.to_dict() if request.form else {}

    user_query = data.get("query") or data.get("question") or ""
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    rag_output, status_code = run_rag_query(user_query, session_id, user_id)
    return jsonify(rag_output), status_code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

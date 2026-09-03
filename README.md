# RAG Assessment — Skeleton (17 code test cases + Azure infra = 21 total)

Domain: **IT Software License Management**. API-key auth, rate limiting,
hardcoded-secrets, and input-length validation have been REMOVED from this
version's scope (per final scope decision). Four new capabilities were
added: query rewriting, context assembly/reordering, observability logging,
and user-scoped conversation memory (personalization).

Environment variables are unchanged: `/etc/environment.d/rag-lab.conf` on
the VM, loaded into `app/config.py` via `os.environ.get(...)`.

## 21 Test Cases

1. Retrieval Recall
2. Distraction / Near-Duplicate Filtering
3. Document Exclusion Filtering (deprecated + PII, combined)
4. Authority Conflict / Contradiction Detection
5. Budget & Overflow
6. Memory Accumulation
7. Memory Staleness
8. Session Isolation
9. Injection Resistance
10. MCP Routing (Tool Surface)
11. Untrusted Tool Output Validation
12. Grounding & Attribution
13. Failure Handling
14. Query Rewriting / Expansion
15. Context Assembly & Ordering (re-rank + lost-in-middle)
16. Production Observability & Efficiency (structured logging + latency)
17. Personalization / Subject Scoping (user_id-bound session ownership)
18. User Prompt Verification (dedicated system-prompt content check)
19-21. Azure infra (Search+Index, Data Source+Indexer, OpenAI deployment)

Step-by-step fix instructions are in the accompanying guide document.

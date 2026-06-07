---
title: SchemeSaathi
emoji: 🇮🇳
colorFrom: orange
colorTo: green
sdk: docker
pinned: true
license: mit
short_description: Agentic RAG over 3400+ Indian government welfare schemes
---

<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/LangGraph-1.0-FF6B6B?style=flat-square">
<img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC143C?style=flat-square">
<img src="https://img.shields.io/badge/BGE--M3-Embeddings-orange?style=flat-square">
<img src="https://img.shields.io/badge/Groq-Llama_3.3_70B-green?style=flat-square">
<img src="https://img.shields.io/badge/Redis-Semantic_Cache-red?style=flat-square&logo=redis&logoColor=white">
<img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square">

</div>

---

# SchemeSaathi 🇮🇳

**Production-grade agentic RAG system that helps Indian citizens discover government welfare schemes they're eligible for.**

India has 3,400+ central and state welfare schemes. Most eligible citizens never find them due to language barriers, complex documentation, and poor discoverability. SchemeSaathi solves this with a stateful AI agent grounded in official government data.

---

## Demo

> Try it live → **[huggingface.co/spaces/RavindranadhhM/schemesaathi](https://huggingface.co/spaces/RavindranadhhM/schemesaathi)**

**Example queries:**
- *"I am a 35-year-old SC farmer in Karnataka with income ₹80,000. What schemes am I eligible for?"*
- *"I am a widow in Maharashtra from a BPL family. What pension schemes exist?"*
- *"BTech student in Bangalore — what scholarships and internship schemes are available?"*

---

## Architecture
User Query
│
├─► Semantic Cache (Redis)  ──── HIT ──► Response (~50ms)
│                                │
│                               MISS
│                                │
├─► Scope Gate (keyword classifier)
│
├─► Profile Parser (city→state mapping, keyword extraction + Groq fallback)
│
├─► Hybrid Retriever
│       ├── Dense search (BGE-M3 cosine similarity, Qdrant)
│       ├── Name-boosted retrieval (exact scheme matching)
│       └── Profile-aware re-scoring (state + Central boost)
│
├─► Grader (vector score threshold 0.55)
│
├─► Generator (Groq Llama-3.3-70B, grounded generation)
│
└─► Memory (session summary compression, Groq Llama-3.1-8B)

---

## Evaluation

Evaluated on a 15-query golden dataset using **Llama-3.3-70B as judge** (no OpenAI dependency):

| Metric | Score |
|---|---|
| Answer Relevancy | **0.95** |
| Context Precision | **0.86** |
| Overall | **0.77** |

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph 1.0 (stateful agent) |
| Vector Store | Qdrant Cloud (27,262 chunks, HNSW index) |
| Embeddings | BGE-M3 — BAAI/bge-m3 (1024-dim, multilingual) |
| LLM | Groq Llama-3.3-70B (generation) + Llama-3.1-8B (memory) |
| Semantic Cache | Redis (cosine similarity, 0.92 threshold, 24h TTL) |
| API | FastAPI + SlowAPI (10 req/min rate limiting) |
| Evaluation | RAGAS-style with open-source LLM judge |
| Data | 3,400 schemes from myscheme.gov.in (NDSAP licensed) |

---

## Data Pipeline
myscheme.gov.in (3,400 schemes)
│
├─► Hierarchical chunker
│       ├── Summary chunks    (~200 tokens, parent)
│       ├── Section chunks    (~400 tokens, eligibility/benefits/documents/process)
│       └── Fact chunks       (~80 tokens, numeric facts)
│
├─► BGE-M3 embeddings (27,262 chunks × 1024 dims)
│
└─► Qdrant (4 payload indexes: level, chunk_type, scheme_id, parent_id)

---

## Key Engineering Decisions

**Metadata-first retrieval** — Every chunk carries 15 metadata fields (state, level, category, tags, chunk_type). Pre-filtering cuts search space ~85% before vector search.

**Profile-aware re-scoring** — Central schemes get +0.05 score boost (apply to all citizens). Matching state schemes get +0.15. Other-state schemes get -0.10 penalty.

**1 LLM call per query** — Scope gate (keyword), grader (vector scores), profile parser (regex + keyword) all run without LLM. Only the generator uses Groq — making the system efficient on free-tier quotas.

**Semantic caching** — Query embeddings cached in Redis with cosine similarity matching. Cache hits return in ~50ms vs ~25s for full pipeline.

---

## Project Structure
schemesaathi/
├── ingestor/          # Data pipeline (loader, cleaner, chunker, embedder, Qdrant uploader)
├── agent/
│   ├── graph.py       # LangGraph StateGraph definition
│   ├── state.py       # AgentState TypedDict
│   ├── prompts.py     # All prompts (versioned)
│   └── nodes/         # scope_gate, profile_parser, retriever, grader, generator, memory
├── api/               # FastAPI app + routes + rate limiting
├── eval/              # RAGAS-style evaluation pipeline
└── frontend/          # Single-file HTML/CSS/JS UI

---

## Legal

Data sourced from [myscheme.gov.in](https://myscheme.gov.in) under the National Data Sharing and Accessibility Policy (NDSAP). For informational purposes only. Verify eligibility at official government portals before applying.

---

<div align="center">
Built by <a href="https://github.com/RavindranadhhM">Ravindranadh M</a> · REVA University, Bengaluru
</div>

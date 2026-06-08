---
title: SchemeSaathi
emoji: 🏛️
colorFrom: yellow
colorTo: green
sdk: docker
pinned: true
license: mit
short_description: Agentic RAG over 3400+ Indian government welfare schemes
---

> **Note for GitHub visitors:** The table above is required configuration for the HuggingFace Spaces deployment. It is not part of the documentation.

---

# SchemeSaathi 🇮🇳

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-FF6B35?style=flat-square)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC143C?style=flat-square)
![BGE-M3](https://img.shields.io/badge/BGE--M3-Embeddings-FF8C00?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-00A67E?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-Semantic_Cache-DC382D?style=flat-square&logo=redis&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-F7DF1E?style=flat-square)

<br>

**Production-grade agentic RAG system for discovering Indian government welfare schemes.**

[**🚀 Live Demo**](https://huggingface.co/spaces/RavindranadhM/schemesaathi) &nbsp;·&nbsp; [**📖 GitHub**](https://github.com/RavindranadhhM/schemesaathi)

</div>

---

## Problem Statement

India has **3,400+ central and state welfare schemes** covering agriculture, education, health, housing, employment, and social welfare. Most eligible citizens never find them — due to language barriers, complex documentation, and poor discoverability. SchemeSaathi solves this with a stateful AI agent grounded in official government data.

---

## Live Demo

**→ [huggingface.co/spaces/RavindranadhM/schemesaathi](https://huggingface.co/spaces/RavindranadhM/schemesaathi)**

Example queries to try:

- *"I am a 35-year-old SC farmer in Karnataka with income ₹80,000. What schemes am I eligible for?"*
- *"I am a widow in Maharashtra from a BPL family. What pension schemes exist?"*
- *"BTech student in Bangalore — what scholarships and internship schemes are available?"*
- *"What is PM KISAN and who is eligible?"*

---

## System Architecture

```
User Query (English or Hindi)
          │
          ▼
┌─────────────────────────┐
│    Semantic Cache       │ ──── HIT ──→ Response in ~50ms
│    Redis + cosine 0.92  │
└───────────┬─────────────┘
            │ MISS
            ▼
┌─────────────────────────┐
│      Scope Gate         │  Pure keyword classifier — zero LLM cost
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│     Profile Parser      │  60+ city→state mappings + regex
│                         │  Groq Llama-3.1-8B fallback for Hindi
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│    Hybrid Retriever     │  BGE-M3 dense cosine similarity
│                         │  Name-boosted exact scheme matching
│                         │  Profile-aware re-scoring:
│                         │    Central        +0.05 (all citizens)
│                         │    Matching state +0.15
│                         │    Other state    −0.10
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│       Grader            │  Vector score threshold ≥ 0.55
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Generator — Groq       │  Llama-3.3-70B, grounded with citations
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Memory Node            │  Session summary, Groq Llama-3.1-8B
└───────────┬─────────────┘
            ▼
       Final Response
```

---

## Evaluation

Evaluated on a **15-query golden dataset** using Llama-3.3-70B as judge (no OpenAI dependency):

| Metric | Score |
|---|---|
| Answer Relevancy | **0.95** |
| Context Precision | **0.86** |
| Overall | **0.77** |

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph 1.0 — stateful graph, 6 nodes |
| Vector Store | Qdrant Cloud — 27,262 chunks, HNSW index, 4 payload indexes |
| Embeddings | BGE-M3 (BAAI/bge-m3) — 1024-dim dense, Hindi + English |
| LLM Generation | Groq Llama-3.3-70B |
| LLM Memory / Parsing | Groq Llama-3.1-8B |
| Semantic Cache | Redis — cosine similarity, 0.92 threshold, 24h TTL |
| API | FastAPI + SlowAPI — async, 10 req/min per IP |
| Data | 3,400 schemes from myscheme.gov.in (NDSAP licensed) |

---

## Data Pipeline

```
myscheme.gov.in CSV (3,400 schemes)
        │
        ▼
Hierarchical Chunker
  ├── Summary chunks     ~200 tokens  ← parent chunk, broad retrieval
  ├── Section chunks     ~400 tokens  ← eligibility / benefits / documents / process
  └── Fact chunks        ~80 tokens   ← numeric facts, amounts, age limits
        │
        ▼
BGE-M3 Embeddings — 27,262 chunks × 1024 dimensions
        │
        ▼
Qdrant Vector Store
  └── Payload indexes: level · chunk_type · scheme_id · parent_id
```

---

## Key Engineering Decisions

**Metadata-first retrieval** — Every chunk carries 15 metadata fields. Hard pre-filtering by state and level cuts the search space ~85% before vector search runs.

**Profile-aware re-scoring** — Central schemes +0.05 (apply to all citizens). Matching-state schemes +0.15. Other-state schemes −0.10. Eliminates irrelevant state-specific results.

**1 LLM call per query** — Scope gate (keyword), grader (vector threshold), and profile parser (regex + city mapping) run without LLM. Only the generator uses Groq — efficient on free-tier quotas.

**City-to-state resolution** — 60+ Indian cities mapped to states at query time. "Bangalore" → Karnataka. Improves retrieval without extra API calls.

**Semantic caching** — Query embeddings cached in Redis with cosine similarity lookup. Cache hits ~50ms vs ~25s full pipeline.

---

## Project Structure

```
schemesaathi/
├── ingestor/
│   ├── data_loader.py        CSV → structured scheme dicts
│   ├── cleaner.py            Text normalisation
│   ├── chunker.py            Hierarchical chunking (summary/section/fact)
│   ├── embedder.py           BGE-M3 dense embeddings
│   └── qdrant_loader.py      Upload with metadata payload indexes
│
├── agent/
│   ├── graph.py              LangGraph StateGraph definition
│   ├── state.py              AgentState TypedDict
│   ├── prompts.py            All prompts
│   └── nodes/
│       ├── cache_check.py    Redis semantic cache lookup
│       ├── scope_gate.py     Keyword scope classifier
│       ├── profile_parser.py City-to-state + keyword extraction
│       ├── retriever.py      Hybrid retrieval + profile re-scoring
│       ├── grader.py         Vector score threshold filter
│       ├── generator.py      Groq Llama-3.3-70B generation
│       └── memory.py         Session summary compression
│
├── api/
│   ├── main.py               FastAPI app + CORS + static files
│   └── routes/
│       ├── query.py          POST /query — rate limited
│       └── health.py         GET /health
│
├── eval/
│   ├── ragas_eval.py         Evaluation pipeline
│   ├── score_responses.py    LLM-as-judge scoring
│   └── results/              RAGAS scores JSON
│
└── frontend/
    └── index.html            Single-file production UI
```

---

## Legal

Data sourced from [myscheme.gov.in](https://myscheme.gov.in) under the **National Data Sharing and Accessibility Policy (NDSAP)**. For informational purposes only — verify eligibility at official government portals before applying.

---

<div align="center">

Built by [Ravindranadh M](https://github.com/RavindranadhhM) &nbsp;·&nbsp; B.Tech Robotics & Automation &nbsp;·&nbsp; REVA University, Bengaluru

</div>

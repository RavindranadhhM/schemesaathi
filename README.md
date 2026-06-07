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

# SchemeSaathi 🇮🇳

<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/LangGraph-1.0-FF6B35?style=flat-square">
<img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC143C?style=flat-square">
<img src="https://img.shields.io/badge/BGE--M3-Embeddings-FF8C00?style=flat-square">
<img src="https://img.shields.io/badge/Groq-Llama_3.3_70B-00A67E?style=flat-square">
<img src="https://img.shields.io/badge/Redis-Semantic_Cache-DC382D?style=flat-square&logo=redis&logoColor=white">
<img src="https://img.shields.io/badge/License-MIT-F7DF1E?style=flat-square">

<br><br>

**Production-grade agentic RAG system for discovering Indian government welfare schemes.**

[Live Demo](https://huggingface.co/spaces/RavindranadhM/schemesaathi) · [GitHub](https://github.com/RavindranadhhM/schemesaathi)

> *Note: The metadata block above is required configuration for HuggingFace Spaces.*

</div>

---

## Problem

India has **3,400+ central and state welfare schemes**. Most eligible citizens never find them — due to language barriers, complex documentation, and poor discoverability. SchemeSaathi solves this with a stateful AI agent grounded in official government data.

---

## Live Demo

Try these at [huggingface.co/spaces/RavindranadhM/schemesaathi](https://huggingface.co/spaces/RavindranadhM/schemesaathi):

- *"I am a 35-year-old SC farmer in Karnataka with income ₹80,000. What schemes am I eligible for?"*
- *"I am a widow in Maharashtra from a BPL family. What pension schemes exist?"*
- *"BTech student in Bangalore — what scholarships and internship schemes are available?"*

---

## Architecture
User Query (EN / HI)
│
▼
┌───────────────────────┐
│   Semantic Cache      │──── HIT ──── Response (~50ms)
│   Redis cosine 0.92   │
└───────────┬───────────┘
│ MISS
▼
┌───────────────────────┐
│     Scope Gate        │  keyword classifier, zero LLM cost
└───────────┬───────────┘
▼
┌───────────────────────┐
│    Profile Parser     │  city→state, regex keywords, Groq fallback
└───────────┬───────────┘
▼
┌───────────────────────┐
│   Hybrid Retriever    │  BGE-M3 dense + name-boosted + state re-scoring
└───────────┬───────────┘
▼
┌───────────────────────┐
│       Grader          │  vector score threshold 0.55
└───────────┬───────────┘
▼
┌───────────────────────┐
│  Generator (Groq)     │  Llama-3.3-70B, grounded generation
└───────────┬───────────┘
▼
┌───────────────────────┐
│  Memory (Groq)        │  session summary, Llama-3.1-8B
└───────────┬───────────┘
▼
Response

---

## Evaluation

15-query golden dataset · Llama-3.3-70B as judge · no OpenAI dependency

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
| Vector Store | Qdrant Cloud — 27,262 chunks, HNSW index |
| Embeddings | BGE-M3 — 1024-dim, Hindi + English |
| LLM Generation | Groq Llama-3.3-70B |
| LLM Memory | Groq Llama-3.1-8B |
| Semantic Cache | Redis — cosine sim, 0.92 threshold, 24h TTL |
| API | FastAPI + SlowAPI — async, 10 req/min rate limiting |
| Data | 3,400 schemes — myscheme.gov.in (NDSAP licensed) |

---

## Key Engineering Decisions

**Metadata-first retrieval** — 15 metadata fields per chunk. Pre-filtering cuts search space ~85% before vector search.

**Profile-aware re-scoring** — Central schemes +0.05. Matching-state schemes +0.15. Other-state −0.10.

**1 LLM call per query** — Scope gate, grader, and profile parser run without LLM. Only generator uses Groq.

**City-to-state resolution** — 60+ Indian cities mapped to states. "Bangalore" → Karnataka improves retrieval without extra LLM calls.

**Semantic caching** — Cache hits ~50ms vs ~25s full pipeline.

---

## Project Structure
schemesaathi/
├── ingestor/        data pipeline (loader · cleaner · chunker · embedder · qdrant)
├── agent/
│   ├── graph.py     LangGraph StateGraph
│   ├── state.py     AgentState TypedDict
│   ├── prompts.py   all prompts
│   └── nodes/       scope_gate · profile_parser · retriever · grader · generator · memory
├── api/             FastAPI app + routes
├── eval/            RAGAS-style evaluation pipeline
└── frontend/        single-file production UI

---

## Legal

Data from [myscheme.gov.in](https://myscheme.gov.in) under NDSAP. Informational purposes only — verify at official portals before applying.

---

<div align="center">
Built by <a href="https://github.com/RavindranadhhM">Ravindranadh M</a> · B.Tech Robotics & Automation · REVA University, Bengaluru
</div>

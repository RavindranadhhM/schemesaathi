---
title: SchemeSaathi
emoji: 🇮🇳
colorFrom: orange
colorTo: green
sdk: docker
pinned: true
license: mit
short_description: Agentic RAG over 3400 Indian government welfare schemes
---

# SchemeSaathi 🇮🇳

**Find every Indian government scheme you're eligible for — powered by Agentic RAG**

## What it does
Describe your profile (state, income, caste, age, occupation) and SchemeSaathi searches across 3,400+ central and state government welfare schemes to find what you qualify for — grounded in official sources with zero hallucination.

## Architecture
- **LangGraph** stateful agent with scope gate, profile parser, retriever, grader, generator, memory nodes
- **Qdrant** vector store — 27,262 chunks with hierarchical metadata (summary → section → fact)
- **BGE-M3** multilingual embeddings (Hindi + English, 1024-dim dense vectors)
- **Groq Llama-3.3-70B** for grounded response generation
- **Redis** semantic cache — cosine similarity threshold 0.92, ~50ms cache hits
- **FastAPI** backend with SlowAPI rate limiting

## Evaluation
Evaluated on 15-query golden dataset (Llama-3.3-70B as judge):
- Answer Relevancy: **0.95**
- Context Precision: **0.86**
- Overall: **0.77**

## Tech Stack
`LangGraph` `Qdrant` `BGE-M3` `Groq` `FastAPI` `Redis` `Python 3.11`

## Data Source
3,400 schemes scraped from [myscheme.gov.in](https://myscheme.gov.in) (NDSAP licensed).
For informational purposes only — verify eligibility at official government portals before applying.

## GitHub
[github.com/RavindranadhhM/schemesaathi](https://github.com/RavindranadhhM/schemesaathi)

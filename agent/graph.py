# agent/graph.py
"""
LangGraph stateful agent.
Flow:
  cache_check → (hit) → END
              → (miss) → scope_gate → (out) → END
                                    → (in)  → profile_parser
                                             → retriever
                                             → grader
                                             → generator
                                             → memory → END
"""
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import refinery
from agent.nodes import (
    cache_check, scope_gate, profile_parser,
    retriever, grader, generator, memory,
)


def detect_language(text: str) -> str:
    """Simple heuristic — check for Devanagari characters."""
    return "hi" if any("\u0900" <= c <= "\u097F" for c in text) else "en"


def embed_query(query: str) -> list[float]:
    """Embed the user query using BGE-M3."""
    from FlagEmbedding import BGEM3FlagModel
    if not hasattr(embed_query, "_model"):
        import torch
        if torch.backends.mps.is_available():
            device, fp16 = "mps", True
        elif torch.cuda.is_available():
            device, fp16 = "cuda", True
        else:
            device, fp16 = "cpu", False
        embed_query._model = BGEM3FlagModel(
            "BAAI/bge-m3", use_fp16=fp16, device=device
        )
    out = embed_query._model.encode(
        [query],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return out["dense_vecs"][0].tolist()


# ── Node functions ────────────────────────────────────────────────────────

def node_cache_check(state: AgentState) -> AgentState:
    qe = embed_query(state["raw_query"])
    state["_query_embedding"] = qe          # stash for retriever
    return cache_check.run(state, qe)


def node_scope_gate(state: AgentState) -> AgentState:
    return scope_gate.run(state)


def node_profile_parser(state: AgentState) -> AgentState:
    return profile_parser.run(state)


def node_retriever(state: AgentState) -> AgentState:
    qe = state.get("_query_embedding") or embed_query(state["raw_query"])
    return retriever.run(state, qe)


def node_grader(state: AgentState) -> AgentState:
    return grader.run(state)


def node_generator(state: AgentState) -> AgentState:
    return generator.run(state)


def node_refinery(state: AgentState) -> AgentState:
    return refinery.refine(state)

def node_memory(state: AgentState) -> AgentState:
    return memory.update(state)


# ── Routing functions ─────────────────────────────────────────────────────

def route_cache(state: AgentState) -> str:
    return "end" if state.get("cache_hit") else "scope_gate"


def route_scope(state: AgentState) -> str:
    return "end" if not state.get("is_in_scope") else "profile_parser"


# ── Build graph ───────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("cache_check",    node_cache_check)
    g.add_node("scope_gate",     node_scope_gate)
    g.add_node("profile_parser", node_profile_parser)
    g.add_node("retriever",      node_retriever)
    g.add_node("grader",         node_grader)
    g.add_node("generator",      node_generator)
    g.add_node("refinery",        node_refinery)
    g.add_node("memory",         node_memory)

    g.set_entry_point("cache_check")

    g.add_conditional_edges("cache_check", route_cache, {
        "end":        END,
        "scope_gate": "scope_gate",
    })
    g.add_conditional_edges("scope_gate", route_scope, {
        "end":          END,
        "profile_parser": "profile_parser",
    })
    g.add_edge("profile_parser", "retriever")
    g.add_edge("retriever",      "grader")
    g.add_edge("grader",         "generator")
    g.add_edge("generator",      "refinery")
    g.add_edge("refinery",       "memory")
    g.add_edge("memory",         END)

    return g.compile()


# Singleton
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, session_state: dict | None = None) -> dict:
    """Main entry point. Call this from the API."""
    import time
    from agent.tracer import trace_query
    graph = get_graph()
    _start = time.time()

    initial: AgentState = {
        "raw_query":        query,
        "language":         detect_language(query),
        "user_profile":     None,
        "cache_hit":        False,
        "is_in_scope":      True,
        "latency_tier":     "standard",
        "retrieved_chunks": [],
        "reranked_chunks":  [],
        "grader_scores":    [],
        "response":         "",
        "matched_schemes":  [],
        "citations":        [],
        "validation_passed": False,
        "refined_output":    None,
        "disclaimer":         "",
        "error":            None,
        "session_history":  [],
        "session_summary":  "",
        **(session_state or {}),
    }

    result = graph.invoke(initial)
    latency_ms = (time.time() - _start) * 1000
    trace_query(query, result, latency_ms)
    return result

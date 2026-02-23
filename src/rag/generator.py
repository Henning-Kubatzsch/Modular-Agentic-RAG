# src/rag/generator.py
"""
Local LLM wrapper (llama-cpp) that conforms to our public interfaces.

- Defines an abstract ChatModel Protocol (interface)
- Implements LocalLLM that satisfies ChatModel
- Keeps YAML-driven config loading
- Provides both streaming and non-streaming chat calls
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Literal

import yaml
from llama_cpp import Llama
from pydantic import BaseModel, Field

from rag.interfaces import ChatModel


# =========================
# Config model
# =========================


class LLMConfig(BaseModel):
    model_path: str
    family: Literal["qwen", "qwen2", "qwen2.5", "llama3", "phi3", "mistral"] = "qwen"
    n_ctx: int = Field(default=4096, ge=512, le=32768)
    n_gpu_layers: int = Field(default=-1)
    n_threads: Optional[int] = None
    seed: int = Field(default=42)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repeat_penalty: float = Field(default=1.1, ge=0.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stop: Optional[List[str]] = Field(default=None)
    n_batch: Optional[int] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None

def load_llm_config(path: str) -> LLMConfig:
    """Read YAML and construct LLMConfig."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg.get("llm", {})
    return LLMConfig(**llm_cfg)


# =========================
# Helpers
# =========================

def family_to_chat_format_and_stops(family: str) -> Dict[str, Any]:
    """
    Map a 'family' string to llama.cpp chat_format + default stop tokens.
    Keep names aligned with llama.cpp's built-ins.
    """
    f = family.lower()
    if f in ("qwen", "qwen2", "qwen2.5"):
        # If your GGUF is Qwen2.x, "qwen2" chat_format also works.
        # this statement makes answers very short and just cuts them off
        # return {"chat_format": "qwen", "extra_stops": ["<|im_end|>","\n\n","\n- ","\n1. ","\n•","<|endoftext|>"]}
        return {"chat_format": "qwen", "extra_stops": ["<|im_end|>"]}
    if f in ("llama3", "llama-3"):
        return {"chat_format": "llama-3", "extra_stops": ["<|eot_id|>", "<|end_of_text|>"]}
    if f in ("phi3", "phi-3", "phi3-mini"):
        return {"chat_format": "phi3", "extra_stops": ["<|end|>", "<|endoftext|>"]}
    if f in ("mistral", "mistral-instruct"):
        return {"chat_format": "mistral-instruct", "extra_stops": ["</s>"]}
    # Fallback: raw completion (not recommended)
    return {"chat_format": None, "extra_stops": []}


# =========================
# Implementation
# =========================

class LocalLLM(ChatModel):
    """llama.cpp-backed chat model implementing the ChatModel interface."""

    def __init__(self, cfg: LLMConfig):
        assert os.path.exists(cfg.model_path), f"Model not found: {cfg.model_path}"
        mapping = family_to_chat_format_and_stops(cfg.family)

        # Merge user-provided stops with family defaults
        stops = list(mapping["extra_stops"])
        if cfg.stop:
            stops.extend(s for s in cfg.stop if s not in stops)

        llama_kwargs: Dict[str, Any] = dict(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads or os.cpu_count(),
            n_gpu_layers=cfg.n_gpu_layers,     # -1 = try offloading all to GPU (Metal)
            seed=cfg.seed,
            logits_all=False,                  # reduce memory
            verbose=False,
            chat_format=mapping["chat_format"],
        )
        # Optional perf knobs if provided in config
        if cfg.n_batch is not None:
            llama_kwargs["n_batch"] = cfg.n_batch
        if cfg.use_mmap is not None:
            llama_kwargs["use_mmap"] = cfg.use_mmap
        if cfg.use_mlock is not None:
            llama_kwargs["use_mlock"] = cfg.use_mlock

        self.llama = Llama(**llama_kwargs)
        self.cfg = cfg
        self.stop = stops
        self.chat_format = mapping["chat_format"]

    # ----- ChatModel interface -----

    def make_messages(
        self,
        user: str,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI-style messages. llama.cpp will apply the proper chat template
        for the selected family (via chat_format).
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        return messages

     
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        params = dict(
            temperature=self.cfg.temperature if temperature is None else temperature,
            top_p=self.cfg.top_p if top_p is None else top_p,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
            repeat_penalty=self.cfg.repeat_penalty,
            stop=self.stop,
            stream=False,
        )
        out = self.llama.create_chat_completion(messages=messages, **params)

        # Case 1: non-streaming API returns a dict
        if isinstance(out, dict):
            return out["choices"][0]["message"]["content"]

        # Case 2: defensive fallback – API returned a generator/iterator even with stream=False
        parts: List[str] = []
        for chunk in out:
            choice = chunk["choices"][0]
            # some backends use "delta", others place full message payload
            content = ""
            if "delta" in choice:
                content = choice["delta"].get("content", "") or ""
            elif "message" in choice:
                content = choice["message"].get("content", "") or ""
            else:
                content = choice.get("text", "") or ""
            if content:
                parts.append(content)
        return "".join(parts)


    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        params = dict(
            temperature=self.cfg.temperature if temperature is None else temperature,
            top_p=self.cfg.top_p if top_p is None else top_p,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
            repeat_penalty=self.cfg.repeat_penalty,
            stop=self.stop,
            stream=True,
        )
        stream = self.llama.create_chat_completion(messages=messages, **params)
        for part in stream:
            delta = part["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta


# =========================
# Convenience helper
# =========================

def simple_answer(question: str, system: Optional[str], cfg_path: str) -> str:
    """
    Minimal convenience function to:
    - load config
    - build a LocalLLM (ChatModel)
    - format messages
    - return a single-shot answer
    """
    cfg = load_llm_config(cfg_path)
    llm: ChatModel = LocalLLM(cfg)
    messages = llm.make_messages(user=question, system=system)
    return llm.chat(messages)

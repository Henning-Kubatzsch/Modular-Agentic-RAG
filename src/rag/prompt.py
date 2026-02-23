# src/rag/prompt.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Iterable, List, Dict, Tuple, Literal, Optional
from pydantic import BaseModel, Field
import re
import textwrap

# ----------------------------
# Public API (you use this)
# ----------------------------

Language = Literal["en", "de"]
Style = Literal["steps", "qa"]


class PromptOptions(BaseModel):
    language: Literal["en", "de"] = "en"
    style: Literal["steps", "qa"] = "steps"
    max_context_chars: int = Field(default=4000, ge=500, le=10000)
    cite: bool = True
    require_citations: bool = True

@dataclass
class PromptOptionsOverride:
    language: Optional[Language] = None
    style: Optional[Style] = None
    max_context_chars: Optional[int] = None
    cite: Optional[bool] = None
    require_citations: Optional[bool] = None

def merge_prompt_options(base: PromptOptions, o: Optional[PromptOptionsOverride]) -> PromptOptions:
    if o is None:
        return base
    return PromptOptions(
        language = o.language if o.language is not None else base.language,
        style = o.style if o.style is not None else base.style,
        max_context_chars= o.max_context_chars if o.max_context_chars is not None else base.max_context_chars,
        cite = o.cite if o.cite is not None else base.cite,
        require_citations = o.require_citations if o.require_citations is not None else base.require_citations
    )


def build_prompts(
    question: str,
    hits: List[Dict],
    opts: PromptOptions | None = None
) -> Tuple[str, str]:
    """
    Returns (system, user) strings. `hits` are the retriever documents:
    [{'text': str, 'meta': {...}, 'score': float}, ...]
    """

      # 1) Context + bibliography
    context, bib = _build_context_and_bib(hits, max_chars=opts.max_context_chars)

    # 2) System message
    if opts.cite == False:
        system = _system_prompt(opts)
        user = f"Context:\n{context}\n\nQuestion:\n{question}\n\n" \
               f"Answer rules:\n{_answer_rules(opts)}"
    else:
        system = _system_prompt_for_citing(opts, bib)
        user = f"Context:\n{context}\n\nQuestion:\n{question}\n\n" \
               f"Answer rules:\n{_answer_rules(opts, cite=False)}"
        
    return system, user


def postprocess_answer(
    answer: str,
    num_sources: int,
    opts: PromptOptions | None = None
) -> str:
    """
    Cleans up typical artifacts and ensures formatting & citations are valid.
    """
    opts = opts or PromptOptions()
    a = _normalize(answer)

    # Remove LLM markup/stops
    a = re.sub(r"</s>|<\|endoftext\|>|\[/?assistant\]|\[/?user\]", "", a, flags=re.I)

    # Enforce list (when using steps style)
    if opts.style == "steps" and not re.search(r"^\s*1[.)]", a, flags=re.M):
        print("septs is selected")
        a = _force_numbered_list(a)

    # Clean citations: only allow [1..num_sources]
    if opts.cite:
        a = _clamp_citations(a, max_n=max(1, num_sources))

    # If citations are required but none present → append note
    #if opts.cite and opts.require_citations and not re.search(r"\[\d+]", a):
    #    a += "\n\nNote: No specific source index was cited. Verify with context."

    # Remove noise & trim
    a = a.strip()
    return a if a else "I'm unsure. Please consult a medical professional."

# ----------------------------
# Internals (helper functions)
# ----------------------------

def _build_context_and_bib(hits: List[Dict], max_chars: int) -> Tuple[str, str]:
    """
    Trims context to a character budget and builds a short bibliography.
    """
    print("-"*80)
    print(f"\n\nhits: \n\n{hits}\n\n")

    cleaned = []
    total = 0
    for i, h in enumerate(hits, start=1):    
        txt = _squash(h.get("text", ""), hard_trim=1200)
        # print(f"iteration {i}, txt: {txt}")
        entry = f"[{i}] {txt}"
        if total + len(entry) > max_chars:
            break
        cleaned.append(entry)
        total += len(entry)
    context = "\n\n".join(cleaned)

    print(f"\n\ncontext: \n\n{context}\n\n")

    # print(f"\n\ncontext in _build_context: {context}\n\n")

    bib_lines = []
    for i, h in enumerate(hits, start=1):
        m = h.get("meta", {}) or {}
        title = m.get("title") or m.get("doc") or m.get("source") or "Unknown"
        sec = m.get("section") or m.get("page") or ""
        year = m.get("year") or ""
        id_string = m.get("id") or ""

        bib_lines.append(f"[{i}]")
        if not m:
            bib_lines.append("no meta data")
        else:
            if title:
                bib_lines.append(f" title: {title}")
            if sec:
                bib_lines.append(f" sec: {sec}")
            if year:
                bib_lines.append(f" year: {year}")
            if id_string:
                bib_lines.append(f" id: {id_string}")
    bib = "\n".join(bib_lines)
    return context, bib

def _system_prompt_for_citing(opts: PromptOptions, bib: str) -> str:
    if opts.language == "de":
        rules = textwrap.dedent(
        f"""

       Du bist ein präziser, sicherheitsorientierter Assistent, der sich an ERC/Erste-Hilfe-Richtlinien hält.
        Antworte kurz und korrekt mit klaren Schritten. Bei Unsicherheit weise explizit darauf hin.
        Falls Kontextquellen vorliegen, zitiere sie als [1], [2], … passend zur Quellenliste.
        Quellen:

        {bib}

        """).strip()
    else:
        rules = textwrap.dedent(
        f"""

        You are a concise, safety-first assistant aligned with ERC/first-aid guidelines.
        Answer briefly and correctly with clear steps. If unsure, say so explicitly.
        When context citations exist, cite [1], [2], … matching the source list.
        Sources:

        {bib}

        """).strip()
    return rules

def _system_prompt(opts: PromptOptions) -> str:
    if opts.language == "de":
        rules = textwrap.dedent(
        f"""
        
        Du bist ein pädagogischer Agent für Erste Hilfe.

        """).strip()
    else:
        rules = textwrap.dedent(
        f"""

        You are a padagogicle Agent in a First Aid situation.

        """).strip()
    return rules

def _answer_rules(opts: PromptOptions, cite: bool = True) -> str:
    if opts.language == "de":
        base = [
            #"Use short, numbered steps (1., 2., 3.).",
            #"Be precise and safety-first.",
            #"If unsure, say: 'I'm unsure.'",
            "Antworte in EXAKT ZWEI SÄTZEN in einem paragraph", 
            "Gib eine kurze Zusammenfassung des gegebenen Kontext",
            "Eine einfache Aufgabe",
            "keine Listen, Aufzählungen, Stichpunkte, Zeilenumbrüche, Überschriften"
        ]
        if opts.cite:
            base.append("Cite sources using [n] that refer to the numbered context chunks.")
    else:
        base = [
            #"Use short, numbered steps (1., 2., 3.).",
            #"Be precise and safety-first.",
            #"If unsure, say: 'I'm unsure.'",
            "Output EXACTLY TWO SENTENCES in one paragraph"
            "Give a brief summary of the given context.",
            "One simple task.",
            "no lists, numbering, bullets, line breaks, or headings"
        ]
        if opts.cite:
            base.append("Cite sources using [n] that refer to the numbered context chunks.")
    return "- " + "\n- ".join(base)

# TODO: don't need this method as chunker is already normalizing
def _normalize(s: str) -> str:
   s = re.sub("\r\n", "\n", s)
   s = re.sub("\r", "\n", s)
   # collapse excessive blank lines
   s = re.sub(r"\n{3,}", "\n\n", s)
   return s.strip()

def _squash(s: str, hard_trim: int = 1200) -> str:
    # s = _normalize(s)
    if len(s) > hard_trim:
        return s[: hard_trim - 1].rsplit(" ", 1)[0] + "…"
    return s

def _force_numbered_list(s: str) -> str:
    # Split into meaningful sentences/lines and number them
    parts = [p.strip(" -•\t") for p in re.split(r'\.\s+|\n', s) if p.strip()]
    parts = [p for p in parts if len(p) > 0]
    print("-" * 80)
    print(f"\nparts:\n\n{parts}\n\n")
    if not parts:
        return s
    enumerated = "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))
    return enumerated

def _clamp_citations(s: str, max_n: int) -> str:
    # Only allow citations [1..max_n]; drop or reduce others
    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= max_n:
            return f"[{n}]"
        return ""  # drop unknown refs
    return re.sub(r"\[(\d+)]", repl, s)
 
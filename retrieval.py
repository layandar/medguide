"""Retrieval for the medication explainer.

Why this is simple on purpose
-----------------------------
openFDA already returns each drug label split into named sections
(dosage_and_administration, warnings, drug_interactions, ...). So for a demo
we do *not* need embeddings: we look the label up by active ingredient, split
the sections into chunks, and rank the chunks with a small keyword scorer.

Upgrade path: swap `rank_chunks` for pgvector similarity search over the same
chunks once you index more sources (DailyMed, WHO, Jordan FDA directory).

Network note: results are cached on disk in data/cache/, so after one
successful fetch (see prefetch.py) the demo works offline.
"""

import json
import math
import re
from pathlib import Path

import httpx

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
CACHE_DIR = Path(__file__).parent / "data" / "cache"

# Sections we always include if present, in priority order.
CORE_SECTIONS = [
    "indications_and_usage",
    "dosage_and_administration",
    "warnings",
    "warnings_and_cautions",
    "boxed_warning",
    "contraindications",
    "drug_interactions",
    "pregnancy",
    "information_for_patients",
    "stop_use",
    "ask_doctor",
    "do_not_use",
]
# Extra sections that can be pulled in when the person asks a specific question.
OPTIONAL_SECTIONS = ["adverse_reactions", "overdosage", "storage_and_handling", "how_supplied", "pediatric_use", "geriatric_use"]
ACTIVE_FIELDS = ["active_ingredient", "purpose"]

CHUNK_CHARS = 900
MAX_CONTEXT_CHARS = 9000  # keeps the prompt small and cheap


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", name.lower()).strip()


def _cache_path(name: str) -> Path:
    return CACHE_DIR / (re.sub(r"[^a-z0-9]+", "_", _norm(name)) + ".json")


def _first(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value or "")


def fetch_label(name: str, client: httpx.Client | None = None) -> dict | None:
    """Return one openFDA label record for a drug name (generic or brand), or None."""
    cache = _cache_path(name)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    q = _norm(name)
    # In openFDA a space between terms means OR.
    search = f'openfda.generic_name:"{q}" openfda.brand_name:"{q}" openfda.substance_name:"{q}"'
    own = client is None
    client = client or httpx.Client(timeout=15)
    try:
        r = client.get(OPENFDA_URL, params={"search": search, "limit": 10})
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
    except (httpx.HTTPError, ValueError):
        return None
    finally:
        if own:
            client.close()

    # Prefer a record that actually has dosing text, then the longest one.
    results = [x for x in results if x.get("dosage_and_administration") or x.get("indications_and_usage")]
    if not results:
        return None
    best = max(results, key=lambda x: len(_first(x.get("dosage_and_administration"))) + len(_first(x.get("warnings"))))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(best), encoding="utf-8")
    return best


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_to_chunks(label: dict, extra_sections: bool = False) -> list[dict]:
    """Split a label into chunks: [{'id', 'section', 'text'}]."""
    sections = CORE_SECTIONS + (OPTIONAL_SECTIONS if extra_sections else [])
    chunks = []
    for sec in sections + ACTIVE_FIELDS:
        raw = _first(label.get(sec))
        if not raw:
            continue
        text = _clean(raw)
        # split on sentence-ish boundaries to ~CHUNK_CHARS
        buf = ""
        n = 0
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if len(buf) + len(sentence) > CHUNK_CHARS and buf:
                chunks.append({"id": f"{sec}#{n}", "section": sec, "text": buf.strip()})
                n += 1
                buf = ""
            buf += sentence + " "
        if buf.strip():
            chunks.append({"id": f"{sec}#{n}", "section": sec, "text": buf.strip()})
    return chunks


# --- tiny BM25-style scorer (no dependencies) -----------------------------
_STOP = set("the a an and or of to in is are for with on at by be it this that you your can may do not if".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


def rank_chunks(chunks: list[dict], query: str, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    docs = [_tokens(c["text"]) for c in chunks]
    avg = sum(len(d) for d in docs) / max(len(docs), 1)
    n = len(docs)
    q = set(_tokens(query))
    df = {t: sum(1 for d in docs if t in d) for t in q}
    scored = []
    for c, d in zip(chunks, docs):
        score = 0.0
        for t in q:
            tf = d.count(t)
            if not tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / (avg or 1)))
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored if s > 0]


def build_context(label: dict, question: str | None = None) -> tuple[str, list[str]]:
    """Return (context_text, section_names_used).

    1. Always include the core sections (what it's for, dosing, warnings, interactions).
    2. If the person asked a specific question, add the best-matching chunks from
       the optional sections too.
    """
    core = label_to_chunks(label)
    picked = list(core)
    if question and question.strip():
        everything = label_to_chunks(label, extra_sections=True)
        have = {c["id"] for c in picked}
        extra = [c for c in rank_chunks(everything, question)[:4] if c["id"] not in have]
        picked += extra

    parts, used, total = [], [], 0
    # keep core order first, then extras
    for c in picked:
        piece = f"[{c['section']}] {c['text']}"
        if total + len(piece) > MAX_CONTEXT_CHARS:
            continue
        parts.append(piece)
        total += len(piece)
        if c["section"] not in used:
            used.append(c["section"])
    return "\n\n".join(parts), used


def get_label_context(drug_name: str, question: str | None = None) -> dict | None:
    """Main entry: drug name -> {'context', 'sections', 'label'} or None if nothing reliable found."""
    label = fetch_label(drug_name)
    if not label:
        return None
    context, sections = build_context(label, question)
    if len(context) < 200:  # too thin to trust
        return None
    return {"context": context, "sections": sections, "label": label}

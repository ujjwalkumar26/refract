"""Query analyzer — detects query type and computes query features.

Analyzes the query text and/or vector to produce a QueryProfile that
the router uses to determine metric weights.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
from scipy.special import softmax  # type: ignore[import-untyped]

from refract.types import QueryProfile

# ── Code detection patterns ──────────────────────────────────────────────────

_CODE_PATTERNS = [
    r"\bdef\s+\w+",       # Python function
    r"\bclass\s+\w+",     # Python/Java class
    r"\bfunction\s+\w+",  # JavaScript function
    r"\bimport\s+\w+",    # import statements
    r"->",                # type annotations / arrows
    r"::\w+",             # C++ scope / Haskell types
    r"\{.*\}",            # braces (code blocks)
    r";\s*$",             # trailing semicolons
    r"\w+\.\w+\(",        # method calls like obj.method(
    r"if\s*\(.*\)",       # if statements with parens
    r"for\s+\w+\s+in\s+", # Python for loops
    r"return\s+",         # return statements
    r"#include",          # C/C++ includes
    r"console\.log",      # JavaScript
    r"print\(",           # Python print
]

_CODE_RE = re.compile("|".join(_CODE_PATTERNS), re.MULTILINE)

# ── Structured data patterns ─────────────────────────────────────────────────

_STRUCTURED_PATTERNS = [
    r'"\w+":\s*["\d\[\{]',   # JSON key-value
    r"\w+:\s*\w+",           # key: value pairs
    r"^\s*\|.*\|",           # table-like markdown
    r"<\w+>.*</\w+>",       # XML/HTML tags
]

_STRUCTURED_RE = re.compile("|".join(_STRUCTURED_PATTERNS), re.MULTILINE)


def _detect_query_type(
    query_text: str,
) -> Literal["keyword", "natural_language", "code", "structured"]:
    """Detect the type of query from its text content.

    Detection rules (applied in priority order):
    1. **code** — contains code-like tokens (def, class, ->, {}, etc.)
    2. **structured** — matches JSON, key:value, or table patterns
    3. **keyword** — 4 or fewer tokens with no sentence punctuation
    4. **natural_language** — default

    Args:
        query_text: The raw query string.

    Returns:
        One of "keyword", "natural_language", "code", "structured".
    """
    text = query_text.strip()

    if not text:
        return "keyword"

    # Rule 1: Structured data (check BEFORE code — JSON has braces too)
    if _STRUCTURED_RE.search(text):
        return "structured"

    # Rule 2: Code detection
    if _CODE_RE.search(text):
        return "code"

    # Rule 3: Keyword (short, no sentence punctuation)
    tokens = text.split()
    has_sentence_punct = bool(re.search(r"[.!?;]", text))
    if len(tokens) <= 4 and not has_sentence_punct:
        return "keyword"

    # Rule 4: Default
    return "natural_language"


def _compute_entropy(scores: np.ndarray) -> float:
    """Compute Shannon entropy of softmax(scores).

    Low entropy = one candidate dominates = high confidence.
    High entropy = flat scores = ambiguous query.

    Args:
        scores: Raw similarity scores (any range).

    Returns:
        Shannon entropy in nats. Returns 0.0 for empty or single-element inputs.
    """
    if len(scores) <= 1:
        return 0.0

    # Softmax for numerical stability
    probs = softmax(scores)
    # Avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def analyze_query(
    query_text: str | None,
    query_vector: np.ndarray | None,
    candidate_vectors: np.ndarray | None = None,
    candidate_scores: np.ndarray | None = None,
) -> QueryProfile:
    """Analyze a query to produce a QueryProfile.

    Detects query type, computes token count, embedding norm, and
    entropy of candidate scores (if available).

    Args:
        query_text: Raw query string (None if only vectors provided).
        query_vector: Query embedding vector (None if only text provided).
        candidate_vectors: Candidate embeddings (used for entropy if scores not provided).
        candidate_scores: Pre-computed cosine scores (used for entropy).

    Returns:
        A QueryProfile with all features computed.
    """
    # Detect query type
    if query_text is not None:
        query_type = _detect_query_type(query_text)
        token_count = len(query_text.split())
    else:
        query_type = "natural_language"
        token_count = 0

    # Compute embedding norm
    embedding_norm = float(np.linalg.norm(query_vector)) if query_vector is not None else 0.0

    # Compute entropy from candidate scores
    if candidate_scores is not None and len(candidate_scores) > 0:
        entropy = _compute_entropy(candidate_scores)
    elif query_vector is not None and candidate_vectors is not None:
        # Compute quick cosine scores for entropy
        norms = np.linalg.norm(candidate_vectors, axis=1)
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 1e-12:
            scores = (candidate_vectors @ query_vector) / (safe_norms * q_norm)
            entropy = _compute_entropy(scores)
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    return QueryProfile(
        raw=query_text,
        vector=query_vector,
        query_type=query_type,
        token_count=token_count,
        embedding_norm=embedding_norm,
        entropy=entropy,
    )

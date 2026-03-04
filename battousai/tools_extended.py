"""
tools_extended.py — Extended Tool Library
============================================
Additional tools for Battousai agents beyond the base set defined in tools.py.

New tools:
    http_client       — Simulated HTTP client (GET/POST/PUT/DELETE)
    python_repl       — Safe Python expression evaluator (restricted builtins)
    json_processor    — Parse, query (JSONPath-like), and transform JSON
    text_analyzer     — Word count, sentiment (keyword), readability metrics
    vector_store      — In-memory vector similarity store using cosine similarity
    key_value_db      — Persistent KV store with TTL support (dict-backed simulation)
    task_queue        — Priority task queue that agents can push/pop from
    cron_scheduler    — Tick-based cron-like scheduling for recurring agent tasks
    data_pipeline     — Chain multiple tools together in a pipeline

Registration::

    from battousai.tools_extended import register_extended_tools

    # Call after kernel.boot() and register_builtin_tools():
    register_extended_tools(kernel.tools, kernel.filesystem)

Each tool is a plain Python callable that accepts keyword arguments and
returns a result dict.  All tools carry a TOOL_SPEC dict with metadata
that mirrors the ToolSpec dataclass fields for documentation purposes.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from battousai.tools import ToolManager, ToolSpec


# ---------------------------------------------------------------------------
# Shared in-memory state (singleton stores, reset on module reload)
# ---------------------------------------------------------------------------

# vector_store — { collection_name → { id: (vector, metadata) } }
_VECTOR_STORES: Dict[str, Dict[str, Tuple[List[float], Dict[str, Any]]]] = defaultdict(dict)

# key_value_db  — { db_name → { key: (value, expire_tick_or_None) } }
_KV_STORES: Dict[str, Dict[str, Tuple[Any, Optional[int]]]] = defaultdict(dict)
_KV_TICK: int = 0  # Updated by cron_scheduler / tools at call time

# task_queue — { queue_name → sorted list of (priority, seq, task_dict) }
# Priority min-heap: lowest int = highest priority
_TASK_QUEUES: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = defaultdict(list)
_TASK_SEQ: Dict[str, int] = defaultdict(int)  # monotonic sequence per queue

# cron_scheduler — { schedule_name → cron_entry_list }
_CRON_ENTRIES: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_CRON_LAST_RUN: Dict[str, int] = {}  # entry_key → last_run_tick
_CRON_CURRENT_TICK: int = 0


# ---------------------------------------------------------------------------
# Pure-Python cosine similarity (no numpy)
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two vectors, in range [-1.0, 1.0].

    Returns 0.0 for zero-length vectors to avoid division by zero.
    Handles vectors of different lengths by padding the shorter one with zeros.
    """
    if not a or not b:
        return 0.0
    # Pad shorter vector
    if len(a) < len(b):
        a = a + [0.0] * (len(b) - len(a))
    elif len(b) < len(a):
        b = b + [0.0] * (len(a) - len(b))
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# 1. HTTP Client (simulated)
# ---------------------------------------------------------------------------

# Fake URL → response database for the simulated HTTP client
_HTTP_MOCK_DB: Dict[str, Dict[str, Any]] = {
    "https://api.example.com/users": {
        "status": 200,
        "body": {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]},
    },
    "https://api.example.com/status": {
        "status": 200,
        "body": {"status": "ok", "version": "1.0", "uptime": 9999},
    },
    "https://httpbin.org/get": {
        "status": 200,
        "body": {"origin": "127.0.0.1", "url": "https://httpbin.org/get"},
    },
    "https://httpbin.org/post": {
        "status": 200,
        "body": {"json": None, "url": "https://httpbin.org/post"},
    },
}

TOOL_SPEC_HTTP_CLIENT = {
    "name": "http_client",
    "description": (
        "Simulated HTTP client. Supports GET, POST, PUT, DELETE. "
        "Returns status code and response body. "
        "Use for agent-to-service communication (simulated in this prototype)."
    ),
    "parameters": {
        "url":     {"type": "string",  "required": True,  "description": "Target URL"},
        "method":  {"type": "string",  "required": False, "default": "GET",
                    "description": "HTTP method: GET | POST | PUT | DELETE"},
        "body":    {"type": "object",  "required": False, "default": None,
                    "description": "Request body (dict, will be JSON-serialised)"},
        "headers": {"type": "object",  "required": False, "default": None,
                    "description": "Request headers dict"},
        "timeout": {"type": "integer", "required": False, "default": 30,
                    "description": "Request timeout in seconds (simulated)"},
    },
    "returns": {
        "status":   "HTTP status code (int)",
        "body":     "Response body (dict or string)",
        "headers":  "Response headers (dict)",
        "latency_ms": "Simulated latency in milliseconds",
        "simulated": "Always True in this prototype",
    },
}


def _http_client(
    url: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Simulated HTTP client tool.

    Looks up the URL in an internal mock database. Unknown URLs return a
    generic 200 OK response.  POST/PUT/DELETE operations return a 200 with
    an echo of the request body.

    Args:
        url     — Target URL to request.
        method  — HTTP verb (GET | POST | PUT | DELETE).
        body    — Optional request body dict.
        headers — Optional request headers dict.
        timeout — Timeout hint (unused in simulation).

    Returns:
        Dict with status, body, headers, latency_ms, simulated=True.
    """
    method = method.upper()
    url_lower = url.lower()

    # Simulate a small latency
    simulated_latency = 45 + (len(url) % 80)

    if method == "GET":
        mock = _HTTP_MOCK_DB.get(url)
        if mock is None:
            # Generic 200 for unknown GET
            response_body: Any = {
                "message": f"Simulated GET response for {url}",
                "data": [],
            }
            status = 200
        else:
            response_body = mock["body"]
            status = mock["status"]

    elif method in ("POST", "PUT"):
        mock = _HTTP_MOCK_DB.get(url)
        if mock:
            response_body = dict(mock["body"])
            response_body["json"] = body
        else:
            response_body = {"message": f"Simulated {method} accepted", "echo": body}
        status = 200 if method == "POST" else 204

    elif method == "DELETE":
        response_body = {"message": f"Simulated DELETE for {url}", "deleted": True}
        status = 204

    else:
        response_body = {"error": f"Unsupported method: {method}"}
        status = 405

    return {
        "status": status,
        "body": response_body,
        "headers": {
            "content-type": "application/json",
            "x-simulated": "true",
            "x-request-url": url,
        },
        "latency_ms": simulated_latency,
        "simulated": True,
    }


# ---------------------------------------------------------------------------
# 2. Python REPL (restricted eval)
# ---------------------------------------------------------------------------

# Whitelist of safe builtins for the restricted evaluator
_SAFE_BUILTINS: Dict[str, Any] = {
    # Math
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "sorted": sorted, "len": len, "range": range,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
    "isinstance": isinstance, "type": type,
    # Math module constants and functions
    "pi": math.pi, "e": math.e,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
    "log10": math.log10, "floor": math.floor, "ceil": math.ceil,
    "pow": math.pow, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "factorial": math.factorial,
    # Boolean literals
    "True": True, "False": False, "None": None,
}

# Blocklist: patterns that must not appear even after syntax parsing
_BLOCKED_PATTERNS = re.compile(
    r"\b(import|exec|eval|open|getattr|setattr|delattr|"
    r"__import__|__builtins__|__class__|__subclasses__|"
    r"compile|globals|locals|vars|dir|input|print|"
    r"breakpoint|exit|quit)\b"
)

TOOL_SPEC_PYTHON_REPL = {
    "name": "python_repl",
    "description": (
        "Safe Python expression evaluator with restricted builtins. "
        "Supports arithmetic, list comprehensions, and math operations. "
        "Dangerous builtins (import, exec, eval, open, etc.) are blocked."
    ),
    "parameters": {
        "code": {"type": "string", "required": True,
                 "description": "Python expression or statement to evaluate"},
    },
    "returns": {
        "result":  "The evaluated result (as a string representation)",
        "success": "True if evaluation succeeded",
        "error":   "Error message if success=False",
    },
}


def _python_repl(code: str) -> Dict[str, Any]:
    """
    Restricted Python expression evaluator.

    Evaluates the provided code string in a sandbox with only whitelisted
    builtins.  Any access to dangerous functions (import, exec, open…) is
    blocked both by static pattern matching and by restricting ``__builtins__``.

    Args:
        code — Python expression or simple statement to evaluate.

    Returns:
        Dict with result (str), success (bool), error (str if failed).

    Examples::

        _python_repl("2 ** 10")
        # → {"result": "1024", "success": True, "error": ""}

        _python_repl("[x**2 for x in range(5)]")
        # → {"result": "[0, 1, 4, 9, 16]", "success": True, "error": ""}

        _python_repl("__import__('os').system('ls')")
        # → {"result": "", "success": False, "error": "Blocked pattern detected..."}
    """
    # Static check: block dangerous patterns
    if _BLOCKED_PATTERNS.search(code):
        blocked = _BLOCKED_PATTERNS.search(code).group(0)
        return {
            "result": "",
            "success": False,
            "error": f"Blocked pattern detected: {blocked!r}. "
                     f"Use only whitelisted math and collection operations.",
        }

    # Also block string-based escape attempts
    if any(bad in code for bad in ["__", "lambda", "yield", "async", "await"]):
        found = next((b for b in ["__", "lambda", "yield", "async", "await"] if b in code), "")
        return {
            "result": "",
            "success": False,
            "error": f"Disallowed syntax: {found!r}",
        }

    try:
        # Use eval with a clean globals dict
        result = eval(  # noqa: S307
            code,
            {"__builtins__": _SAFE_BUILTINS},
            {},
        )
        return {
            "result": repr(result) if result is not None else "None",
            "success": True,
            "error": "",
        }
    except SyntaxError as exc:
        return {"result": "", "success": False, "error": f"SyntaxError: {exc}"}
    except Exception as exc:
        return {"result": "", "success": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# 3. JSON Processor
# ---------------------------------------------------------------------------

TOOL_SPEC_JSON_PROCESSOR = {
    "name": "json_processor",
    "description": (
        "Parse, query (JSONPath-like dot notation), and transform JSON data. "
        "Supports operations: parse, stringify, query, set, delete, keys, merge."
    ),
    "parameters": {
        "operation": {
            "type": "string", "required": True,
            "description": (
                "Operation to perform: "
                "parse (JSON string→dict), stringify (dict→JSON string), "
                "query (extract a value by dot path), "
                "set (set a value by dot path), "
                "delete (remove a key by dot path), "
                "keys (list top-level keys), "
                "merge (deep-merge two JSON objects)"
            ),
        },
        "data":    {"type": "any",    "required": True,  "description": "Input data (dict or JSON string)"},
        "path":    {"type": "string", "required": False, "description": "Dot-notation path (e.g. 'a.b.c')"},
        "value":   {"type": "any",    "required": False, "description": "Value to set (for 'set' operation)"},
        "data2":   {"type": "any",    "required": False, "description": "Second object for 'merge' operation"},
        "indent":  {"type": "integer","required": False, "default": 2,
                    "description": "Indentation for stringify"},
    },
    "returns": {
        "result":  "The operation result",
        "success": "True if operation succeeded",
        "error":   "Error description if success=False",
    },
}


def _json_path_get(obj: Any, path: str) -> Any:
    """
    Traverse a nested dict/list structure using dot-notation path.

    Examples::
        _json_path_get({"a": {"b": 42}}, "a.b")  # → 42
        _json_path_get({"items": [1, 2, 3]}, "items.1")  # → 2
    """
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key {part!r} not found")
            current = current[part]
        elif isinstance(current, (list, tuple)):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                raise KeyError(f"Index {part!r} out of range or not an integer")
        else:
            raise KeyError(f"Cannot traverse into {type(current).__name__!r} with key {part!r}")
    return current


def _json_path_set(obj: Dict, path: str, value: Any) -> Dict:
    """Set a value at a dot-notation path, creating intermediate dicts."""
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return obj


def _json_path_delete(obj: Dict, path: str) -> Dict:
    """Delete the value at a dot-notation path."""
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return obj
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)
    return obj


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep-merge two dicts; override values take precedence."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _json_processor(
    operation: str,
    data: Any,
    path: Optional[str] = None,
    value: Any = None,
    data2: Any = None,
    indent: int = 2,
) -> Dict[str, Any]:
    """
    Multi-operation JSON processing tool.

    Args:
        operation — One of: parse, stringify, query, set, delete, keys, merge.
        data      — Primary input (dict or JSON string depending on operation).
        path      — Dot-notation access path for query/set/delete.
        value     — Value to assign (set operation).
        data2     — Second object for merge operation.
        indent    — JSON indentation for stringify.

    Returns:
        Dict with result, success, error fields.
    """
    try:
        op = operation.lower().strip()

        if op == "parse":
            if isinstance(data, dict):
                return {"result": data, "success": True, "error": ""}
            parsed = json.loads(str(data))
            return {"result": parsed, "success": True, "error": ""}

        elif op == "stringify":
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass
            result = json.dumps(data, indent=indent, default=str)
            return {"result": result, "success": True, "error": ""}

        elif op == "query":
            if isinstance(data, str):
                data = json.loads(data)
            if not path:
                return {"result": "", "success": False, "error": "path is required for query"}
            result = _json_path_get(data, path)
            return {"result": result, "success": True, "error": ""}

        elif op == "set":
            if isinstance(data, str):
                data = json.loads(data)
            if not path:
                return {"result": "", "success": False, "error": "path is required for set"}
            result = _json_path_set(dict(data), path, value)
            return {"result": result, "success": True, "error": ""}

        elif op == "delete":
            if isinstance(data, str):
                data = json.loads(data)
            if not path:
                return {"result": "", "success": False, "error": "path is required for delete"}
            result = _json_path_delete(dict(data), path)
            return {"result": result, "success": True, "error": ""}

        elif op == "keys":
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                return {"result": list(data.keys()), "success": True, "error": ""}
            return {"result": [], "success": False, "error": "data is not an object"}

        elif op == "merge":
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data2, str):
                data2 = json.loads(data2)
            if data2 is None:
                return {"result": data, "success": True, "error": ""}
            result = _deep_merge(data, data2)
            return {"result": result, "success": True, "error": ""}

        else:
            return {
                "result": "",
                "success": False,
                "error": f"Unknown operation {operation!r}. "
                         f"Supported: parse, stringify, query, set, delete, keys, merge.",
            }

    except Exception as exc:
        return {"result": "", "success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 4. Text Analyzer
# ---------------------------------------------------------------------------

# Simple positive/negative word lists for keyword-based sentiment
_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "positive", "success", "win", "best", "perfect", "love", "happy",
    "joy", "improve", "benefit", "innovative", "efficient", "effective",
    "clear", "easy", "fast", "smart", "powerful", "robust", "reliable",
    "helpful", "progress", "achieve", "solution", "opportunity", "growth",
}
_NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "horrible", "negative", "fail",
    "worst", "hate", "sad", "error", "bug", "crash", "slow", "weak",
    "broken", "difficult", "complex", "problem", "issue", "risk",
    "threat", "danger", "wrong", "false", "invalid", "failed", "loss",
    "decline", "decrease", "obstacle", "limitation",
}

TOOL_SPEC_TEXT_ANALYZER = {
    "name": "text_analyzer",
    "description": (
        "Analyse text for word count, character count, sentence count, "
        "keyword-based sentiment (positive/negative/neutral), "
        "and Flesch-Kincaid readability estimation."
    ),
    "parameters": {
        "text": {"type": "string", "required": True, "description": "Text to analyse"},
    },
    "returns": {
        "word_count":      "Number of words",
        "char_count":      "Number of characters (excluding spaces)",
        "sentence_count":  "Estimated number of sentences",
        "avg_word_length": "Average word length (chars)",
        "sentiment":       "positive | negative | neutral",
        "sentiment_score": "Float in [-1.0, 1.0] (positive = more positive)",
        "positive_words":  "List of positive keywords found",
        "negative_words":  "List of negative keywords found",
        "readability":     "Flesch-Kincaid grade estimate (approximate)",
        "top_words":       "Top 5 most frequent words (excluding stop words)",
    },
}

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "it", "its", "this", "that", "these", "those",
    "as", "by", "from", "into", "not", "no", "so", "if", "then", "than",
}


def _text_analyzer(text: str) -> Dict[str, Any]:
    """
    Analyse a text string and return linguistic metrics.

    Sentiment is determined by counting positive and negative keywords
    from a curated word list.  Readability uses a simplified version of
    the Flesch-Kincaid formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59.
    Syllables are approximated by counting vowel groups.

    Args:
        text — Input text string to analyse.

    Returns:
        Dict with word_count, char_count, sentence_count, avg_word_length,
        sentiment, sentiment_score, positive_words, negative_words,
        readability, top_words.
    """
    if not text or not text.strip():
        return {
            "word_count": 0, "char_count": 0, "sentence_count": 0,
            "avg_word_length": 0.0, "sentiment": "neutral", "sentiment_score": 0.0,
            "positive_words": [], "negative_words": [], "readability": 0.0, "top_words": [],
        }

    # Tokenise
    words_raw = re.findall(r"\b[a-zA-Z']+\b", text)
    words_lower = [w.lower() for w in words_raw]
    word_count = len(words_lower)
    char_count = sum(len(w) for w in words_lower)

    # Sentence count: split on .!?
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(1, len(sentences))

    avg_word_length = (char_count / word_count) if word_count > 0 else 0.0

    # Sentiment
    pos_found = [w for w in words_lower if w in _POSITIVE_WORDS]
    neg_found = [w for w in words_lower if w in _NEGATIVE_WORDS]
    pos_count, neg_count = len(pos_found), len(neg_found)
    total_sentiment = pos_count + neg_count
    if total_sentiment == 0:
        sentiment_score = 0.0
        sentiment = "neutral"
    else:
        sentiment_score = (pos_count - neg_count) / total_sentiment
        if sentiment_score > 0.1:
            sentiment = "positive"
        elif sentiment_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

    # Syllable approximation for readability
    def _syllable_count(word: str) -> int:
        word = word.lower()
        vowels = re.findall(r"[aeiouy]+", word)
        return max(1, len(vowels))

    total_syllables = sum(_syllable_count(w) for w in words_lower)
    # Flesch-Kincaid Grade Level (approximate)
    if word_count > 0 and sentence_count > 0:
        asl = word_count / sentence_count  # avg sentence length
        asw = total_syllables / word_count  # avg syllables per word
        fk_grade = 0.39 * asl + 11.8 * asw - 15.59
        readability = round(max(0.0, fk_grade), 2)
    else:
        readability = 0.0

    # Top words (exclude stop words)
    freq: Dict[str, int] = {}
    for w in words_lower:
        if w not in _STOP_WORDS and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    top_words = sorted(freq, key=lambda w: freq[w], reverse=True)[:5]

    return {
        "word_count":      word_count,
        "char_count":      char_count,
        "sentence_count":  sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "sentiment":       sentiment,
        "sentiment_score": round(sentiment_score, 4),
        "positive_words":  list(set(pos_found)),
        "negative_words":  list(set(neg_found)),
        "readability":     readability,
        "top_words":       top_words,
    }


# ---------------------------------------------------------------------------
# 5. Vector Store
# ---------------------------------------------------------------------------

TOOL_SPEC_VECTOR_STORE = {
    "name": "vector_store",
    "description": (
        "In-memory vector similarity store using cosine similarity (pure Python). "
        "Supports add, search, delete, list, and clear operations. "
        "Useful for semantic search and nearest-neighbour retrieval."
    ),
    "parameters": {
        "operation":   {"type": "string", "required": True,
                        "description": "add | search | delete | list | clear"},
        "collection":  {"type": "string", "required": False, "default": "default",
                        "description": "Name of the vector collection"},
        "id":          {"type": "string", "required": False,
                        "description": "Unique identifier for the vector (add/delete)"},
        "vector":      {"type": "array",  "required": False,
                        "description": "Embedding vector as list of floats (add/search)"},
        "metadata":    {"type": "object", "required": False, "default": {},
                        "description": "Arbitrary metadata stored alongside the vector"},
        "top_k":       {"type": "integer","required": False, "default": 5,
                        "description": "Number of results to return (search)"},
    },
    "returns": {
        "success":  "True if operation succeeded",
        "result":   "Operation-specific result",
        "error":    "Error description if success=False",
    },
}


def _vector_store(
    operation: str,
    collection: str = "default",
    id: Optional[str] = None,
    vector: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    In-memory vector store with cosine similarity search.

    Collections are isolated namespaces within the shared store.
    Vectors persist for the lifetime of the Python process (or until cleared).

    Operations:
        add    — Store a vector with an ID and optional metadata.
        search — Find the top_k most similar vectors to a query vector.
        delete — Remove a vector by ID.
        list   — List all IDs in a collection.
        clear  — Remove all vectors from a collection.

    Args:
        operation  — One of: add | search | delete | list | clear.
        collection — Namespace for the vectors (default: "default").
        id         — Unique key for the vector (required for add/delete).
        vector     — List of floats (required for add/search).
        metadata   — Optional dict stored alongside the vector.
        top_k      — Number of nearest neighbours to return in search.

    Returns:
        Dict with success, result, error.
    """
    op = operation.lower().strip()
    store = _VECTOR_STORES[collection]

    if op == "add":
        if id is None:
            return {"success": False, "result": None, "error": "id is required for add"}
        if vector is None:
            return {"success": False, "result": None, "error": "vector is required for add"}
        store[id] = (list(vector), dict(metadata or {}))
        return {"success": True, "result": {"id": id, "dim": len(vector)}, "error": ""}

    elif op == "search":
        if vector is None:
            return {"success": False, "result": None, "error": "vector is required for search"}
        if not store:
            return {"success": True, "result": [], "error": ""}
        scored: List[Tuple[float, str]] = []
        for vec_id, (stored_vec, meta) in store.items():
            sim = _cosine_similarity(vector, stored_vec)
            scored.append((sim, vec_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            {
                "id": vec_id,
                "score": round(sim, 6),
                "metadata": store[vec_id][1],
            }
            for sim, vec_id in scored[:top_k]
        ]
        return {"success": True, "result": results, "error": ""}

    elif op == "delete":
        if id is None:
            return {"success": False, "result": None, "error": "id is required for delete"}
        removed = store.pop(id, None)
        return {
            "success": True,
            "result": {"deleted": removed is not None, "id": id},
            "error": "",
        }

    elif op == "list":
        return {"success": True, "result": list(store.keys()), "error": ""}

    elif op == "clear":
        count = len(store)
        _VECTOR_STORES[collection] = {}
        return {"success": True, "result": {"cleared": count}, "error": ""}

    else:
        return {
            "success": False,
            "result": None,
            "error": f"Unknown operation {operation!r}. Supported: add, search, delete, list, clear.",
        }


# ---------------------------------------------------------------------------
# 6. Key-Value DB
# ---------------------------------------------------------------------------

TOOL_SPEC_KEY_VALUE_DB = {
    "name": "key_value_db",
    "description": (
        "Persistent (in-process) key-value store with optional TTL support. "
        "Supports get, set, delete, exists, keys, flush, and ttl operations. "
        "Multiple named databases are supported (default: 'default')."
    ),
    "parameters": {
        "operation": {"type": "string",  "required": True,
                      "description": "get | set | delete | exists | keys | flush | ttl"},
        "db":        {"type": "string",  "required": False, "default": "default",
                      "description": "Database namespace"},
        "key":       {"type": "string",  "required": False,
                      "description": "Key to operate on"},
        "value":     {"type": "any",     "required": False,
                      "description": "Value to store (set operation)"},
        "ttl_ticks": {"type": "integer", "required": False, "default": None,
                      "description": "Time-to-live in ticks (None = no expiry)"},
        "current_tick": {"type": "integer", "required": False, "default": 0,
                         "description": "Current system tick for TTL evaluation"},
    },
    "returns": {
        "success": "True if operation succeeded",
        "result":  "Operation-specific result",
        "error":   "Error description if success=False",
    },
}


def _key_value_db(
    operation: str,
    db: str = "default",
    key: Optional[str] = None,
    value: Any = None,
    ttl_ticks: Optional[int] = None,
    current_tick: int = 0,
) -> Dict[str, Any]:
    """
    In-process key-value database with TTL-based expiration.

    Data is stored in a Python dict and persists for the lifetime of the
    process.  TTLs are evaluated lazily on reads.

    Operations:
        get    — Retrieve a value by key (returns None if missing/expired).
        set    — Store a value with optional TTL.
        delete — Remove a key.
        exists — Check if a key exists and has not expired.
        keys   — List all non-expired keys in the database.
        flush  — Remove all keys from the database.
        ttl    — Return the remaining TTL of a key (in ticks).

    Args:
        operation    — Operation to perform.
        db           — Database namespace.
        key          — Key to operate on (most operations).
        value        — Value to store (set).
        ttl_ticks    — Expiry TTL in ticks (set operation; None = no expiry).
        current_tick — Current system tick for TTL evaluation.

    Returns:
        Dict with success, result, error.
    """
    op = operation.lower().strip()
    store = _KV_STORES[db]

    # Lazy expiry helper
    def _is_expired(entry: Tuple[Any, Optional[int]]) -> bool:
        _, expire_at = entry
        return expire_at is not None and current_tick >= expire_at

    if op == "get":
        if key is None:
            return {"success": False, "result": None, "error": "key is required for get"}
        entry = store.get(key)
        if entry is None:
            return {"success": True, "result": None, "error": ""}
        if _is_expired(entry):
            del store[key]
            return {"success": True, "result": None, "error": ""}
        return {"success": True, "result": entry[0], "error": ""}

    elif op == "set":
        if key is None:
            return {"success": False, "result": None, "error": "key is required for set"}
        expire_at = (current_tick + ttl_ticks) if ttl_ticks is not None else None
        store[key] = (value, expire_at)
        return {"success": True, "result": key, "error": ""}

    elif op == "delete":
        if key is None:
            return {"success": False, "result": None, "error": "key is required for delete"}
        removed = store.pop(key, None)
        return {"success": True, "result": removed is not None, "error": ""}

    elif op == "exists":
        if key is None:
            return {"success": False, "result": False, "error": "key is required for exists"}
        entry = store.get(key)
        if entry is None or _is_expired(entry):
            return {"success": True, "result": False, "error": ""}
        return {"success": True, "result": True, "error": ""}

    elif op == "keys":
        live_keys = [k for k, v in store.items() if not _is_expired(v)]
        return {"success": True, "result": live_keys, "error": ""}

    elif op == "flush":
        count = len(store)
        _KV_STORES[db] = {}
        return {"success": True, "result": {"flushed": count}, "error": ""}

    elif op == "ttl":
        if key is None:
            return {"success": False, "result": None, "error": "key is required for ttl"}
        entry = store.get(key)
        if entry is None:
            return {"success": True, "result": -2, "error": ""}  # -2 = does not exist
        _, expire_at = entry
        if expire_at is None:
            return {"success": True, "result": -1, "error": ""}  # -1 = no expiry
        remaining = expire_at - current_tick
        return {"success": True, "result": max(0, remaining), "error": ""}

    else:
        return {
            "success": False,
            "result": None,
            "error": f"Unknown operation {operation!r}. Supported: get, set, delete, exists, keys, flush, ttl.",
        }


# ---------------------------------------------------------------------------
# 7. Task Queue
# ---------------------------------------------------------------------------

TOOL_SPEC_TASK_QUEUE = {
    "name": "task_queue",
    "description": (
        "Priority task queue shared across agents. "
        "Supports push, pop, peek, size, list, and clear operations. "
        "Lower priority number = higher urgency (min-heap semantics)."
    ),
    "parameters": {
        "operation": {"type": "string",  "required": True,
                      "description": "push | pop | peek | size | list | clear"},
        "queue":     {"type": "string",  "required": False, "default": "default",
                      "description": "Queue name/namespace"},
        "task":      {"type": "object",  "required": False,
                      "description": "Task dict to enqueue (push operation)"},
        "priority":  {"type": "integer", "required": False, "default": 5,
                      "description": "Priority level 0-9 (lower = higher urgency)"},
    },
    "returns": {
        "success": "True if operation succeeded",
        "result":  "Operation-specific result",
        "error":   "Error description if success=False",
    },
}


def _task_queue(
    operation: str,
    queue: str = "default",
    task: Optional[Dict[str, Any]] = None,
    priority: int = 5,
) -> Dict[str, Any]:
    """
    Priority task queue for inter-agent work distribution.

    Tasks are stored as (priority, sequence, task_dict) tuples.
    The queue is ordered by priority (ascending), then by insertion
    order (FIFO within the same priority level).

    Operations:
        push  — Add a task with a given priority.
        pop   — Remove and return the highest-priority task.
        peek  — Return the highest-priority task without removing it.
        size  — Return the number of tasks in the queue.
        list  — Return all tasks as a list (sorted by priority).
        clear — Remove all tasks from the queue.

    Args:
        operation — Operation to perform.
        queue     — Queue namespace name.
        task      — Task dict to enqueue (push operation).
        priority  — Task priority 0-9 (push operation).

    Returns:
        Dict with success, result, error.
    """
    import bisect
    op = operation.lower().strip()
    q = _TASK_QUEUES[queue]

    if op == "push":
        if task is None:
            return {"success": False, "result": None, "error": "task is required for push"}
        seq = _TASK_SEQ[queue]
        _TASK_SEQ[queue] += 1
        entry = (priority, seq, task)
        # Insert in sorted position (sorted list as priority queue)
        bisect.insort(q, entry)
        return {
            "success": True,
            "result": {"position": q.index(entry), "queue_size": len(q)},
            "error": "",
        }

    elif op == "pop":
        if not q:
            return {"success": True, "result": None, "error": ""}
        prio, seq, task_data = q.pop(0)
        return {"success": True, "result": {"priority": prio, "task": task_data}, "error": ""}

    elif op == "peek":
        if not q:
            return {"success": True, "result": None, "error": ""}
        prio, seq, task_data = q[0]
        return {"success": True, "result": {"priority": prio, "task": task_data}, "error": ""}

    elif op == "size":
        return {"success": True, "result": len(q), "error": ""}

    elif op == "list":
        tasks = [{"priority": p, "seq": s, "task": t} for p, s, t in q]
        return {"success": True, "result": tasks, "error": ""}

    elif op == "clear":
        count = len(q)
        _TASK_QUEUES[queue] = []
        return {"success": True, "result": {"cleared": count}, "error": ""}

    else:
        return {
            "success": False,
            "result": None,
            "error": f"Unknown operation {operation!r}. Supported: push, pop, peek, size, list, clear.",
        }


# ---------------------------------------------------------------------------
# 8. Cron Scheduler
# ---------------------------------------------------------------------------

TOOL_SPEC_CRON_SCHEDULER = {
    "name": "cron_scheduler",
    "description": (
        "Tick-based cron-like scheduler for recurring agent tasks. "
        "Agents register recurring tasks that auto-execute at fixed tick intervals. "
        "Supports: register, unregister, tick (advance and fire due tasks), list."
    ),
    "parameters": {
        "operation":    {"type": "string",  "required": True,
                         "description": "register | unregister | tick | list"},
        "schedule":     {"type": "string",  "required": False, "default": "default",
                         "description": "Schedule namespace"},
        "name":         {"type": "string",  "required": False,
                         "description": "Unique name for the cron entry (register/unregister)"},
        "every_n_ticks":{"type": "integer", "required": False,
                         "description": "Run every N ticks (register)"},
        "tool":         {"type": "string",  "required": False,
                         "description": "Tool name to invoke when the cron fires"},
        "args":         {"type": "object",  "required": False, "default": {},
                         "description": "Args to pass to the tool on each fire"},
        "current_tick": {"type": "integer", "required": False, "default": 0,
                         "description": "Current system tick (tick operation)"},
    },
    "returns": {
        "success":  "True if operation succeeded",
        "result":   "Operation-specific result (list of fired entries for tick)",
        "error":    "Error description if success=False",
    },
}


def _cron_scheduler(
    operation: str,
    schedule: str = "default",
    name: Optional[str] = None,
    every_n_ticks: Optional[int] = None,
    tool: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    current_tick: int = 0,
) -> Dict[str, Any]:
    """
    Tick-based cron scheduler for registering recurring tool invocations.

    Entries are stored per schedule namespace.  When the ``tick`` operation
    is called with the current tick, any entries whose ``every_n_ticks``
    interval has elapsed since their last run are fired and their results
    are returned.

    Note: the cron scheduler does NOT directly call tools — it returns a
    list of "due" entries that the calling agent should dispatch.  This
    keeps the tool stateless with respect to the kernel.

    Operations:
        register   — Add a recurring entry.
        unregister — Remove an entry by name.
        tick       — Advance the clock; return entries due this tick.
        list       — List all registered entries.

    Args:
        operation    — Operation to perform.
        schedule     — Cron schedule namespace.
        name         — Entry name (register/unregister).
        every_n_ticks — Interval in ticks between executions.
        tool         — Name of the tool to call when entry fires.
        args         — Arguments to pass to the tool.
        current_tick — Current system tick (tick operation).

    Returns:
        Dict with success, result (list of fired entry dicts for tick), error.
    """
    op = operation.lower().strip()
    entries = _CRON_ENTRIES[schedule]

    if op == "register":
        if name is None:
            return {"success": False, "result": None, "error": "name is required for register"}
        if every_n_ticks is None or every_n_ticks < 1:
            return {"success": False, "result": None,
                    "error": "every_n_ticks must be a positive integer"}
        if tool is None:
            return {"success": False, "result": None, "error": "tool is required for register"}
        # Remove existing entry with same name
        _CRON_ENTRIES[schedule] = [e for e in entries if e["name"] != name]
        entry = {
            "name": name,
            "every_n_ticks": every_n_ticks,
            "tool": tool,
            "args": dict(args or {}),
            "registered_tick": current_tick,
            "fire_count": 0,
        }
        _CRON_ENTRIES[schedule].append(entry)
        _CRON_LAST_RUN[f"{schedule}:{name}"] = current_tick
        return {"success": True, "result": entry, "error": ""}

    elif op == "unregister":
        if name is None:
            return {"success": False, "result": None, "error": "name is required for unregister"}
        before = len(_CRON_ENTRIES[schedule])
        _CRON_ENTRIES[schedule] = [e for e in _CRON_ENTRIES[schedule] if e["name"] != name]
        after = len(_CRON_ENTRIES[schedule])
        removed = before - after
        _CRON_LAST_RUN.pop(f"{schedule}:{name}", None)
        return {"success": True, "result": {"removed": removed}, "error": ""}

    elif op == "tick":
        fired: List[Dict[str, Any]] = []
        for entry in _CRON_ENTRIES[schedule]:
            entry_key = f"{schedule}:{entry['name']}"
            last_run = _CRON_LAST_RUN.get(entry_key, entry.get("registered_tick", 0))
            if current_tick - last_run >= entry["every_n_ticks"]:
                entry["fire_count"] += 1
                _CRON_LAST_RUN[entry_key] = current_tick
                fired.append({
                    "name": entry["name"],
                    "tool": entry["tool"],
                    "args": entry["args"],
                    "fire_count": entry["fire_count"],
                    "tick": current_tick,
                })
        return {"success": True, "result": fired, "error": ""}

    elif op == "list":
        snapshot = []
        for entry in _CRON_ENTRIES[schedule]:
            entry_key = f"{schedule}:{entry['name']}"
            last_run = _CRON_LAST_RUN.get(entry_key, entry.get("registered_tick", 0))
            snapshot.append({
                "name": entry["name"],
                "every_n_ticks": entry["every_n_ticks"],
                "tool": entry["tool"],
                "args": entry["args"],
                "fire_count": entry["fire_count"],
                "last_run_tick": last_run,
            })
        return {"success": True, "result": snapshot, "error": ""}

    else:
        return {
            "success": False,
            "result": None,
            "error": f"Unknown operation {operation!r}. Supported: register, unregister, tick, list.",
        }


# ---------------------------------------------------------------------------
# 9. Data Pipeline
# ---------------------------------------------------------------------------

TOOL_SPEC_DATA_PIPELINE = {
    "name": "data_pipeline",
    "description": (
        "Chain multiple tool invocations together in a sequential pipeline. "
        "Each stage's 'result' is injected into the next stage's args as '__input__'. "
        "Useful for composing multi-step data transformations."
    ),
    "parameters": {
        "stages": {
            "type": "array",
            "required": True,
            "description": (
                "Ordered list of pipeline stage dicts. "
                "Each stage: {'tool': 'tool_name', 'args': {}, 'input_key': '__input__'}. "
                "'input_key' specifies which arg key receives the previous stage's result."
            ),
        },
        "initial_input": {
            "type": "any",
            "required": False,
            "default": None,
            "description": "Initial input data passed to the first stage as '__input__'",
        },
    },
    "returns": {
        "success":        "True if all stages completed without error",
        "result":         "Output of the final stage",
        "stage_results":  "List of per-stage results",
        "failed_stage":   "Index of the first failed stage (if success=False)",
        "error":          "Error description if success=False",
    },
}

# Registry of available tools for pipeline use (populated by register_extended_tools)
_PIPELINE_TOOL_REGISTRY: Dict[str, Callable] = {}


def _data_pipeline(
    stages: List[Dict[str, Any]],
    initial_input: Any = None,
) -> Dict[str, Any]:
    """
    Execute a multi-stage data pipeline.

    Each stage specifies a tool name and args.  The ``input_key`` field
    (default: ``"__input__"`` controls which argument key receives the
    previous stage's result.  Stages are executed sequentially; if any
    stage fails, the pipeline halts and reports the failure.

    Example pipeline::

        stages = [
            {"tool": "json_processor",
             "args": {"operation": "parse", "data": '{"name": "Alice"}'},
             "input_key": None},          # No input injection for stage 0
            {"tool": "json_processor",
             "args": {"operation": "query", "path": "name"},
             "input_key": "data"},        # Previous result → args["data"]
            {"tool": "text_analyzer",
             "input_key": "text"},        # Previous result → args["text"]
        ]

    Args:
        stages        — Ordered list of stage configuration dicts.
        initial_input — Initial data fed to the first stage.

    Returns:
        Dict with success, result (last stage output), stage_results, error.
    """
    if not stages:
        return {
            "success": False,
            "result": None,
            "stage_results": [],
            "failed_stage": -1,
            "error": "Pipeline has no stages",
        }

    stage_results: List[Any] = []
    current_input = initial_input

    for idx, stage in enumerate(stages):
        tool_name = stage.get("tool", "")
        stage_args = dict(stage.get("args", {}))
        input_key = stage.get("input_key", "__input__")

        # Inject previous stage's result into this stage's args
        # input_key=None means "don't inject" (useful for the first stage)
        if current_input is not None and input_key is not None and input_key != "":
            stage_args[input_key] = current_input

        # Look up the tool callable
        tool_fn = _PIPELINE_TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            return {
                "success": False,
                "result": None,
                "stage_results": stage_results,
                "failed_stage": idx,
                "error": (
                    f"Stage {idx}: Tool {tool_name!r} not found in pipeline registry. "
                    f"Available: {sorted(_PIPELINE_TOOL_REGISTRY.keys())}"
                ),
            }

        # Execute the tool
        try:
            stage_output = tool_fn(**stage_args)
        except Exception as exc:
            return {
                "success": False,
                "result": None,
                "stage_results": stage_results,
                "failed_stage": idx,
                "error": f"Stage {idx} ({tool_name!r}) raised: {exc}",
            }

        stage_results.append({
            "stage": idx,
            "tool": tool_name,
            "success": stage_output.get("success", True) if isinstance(stage_output, dict) else True,
            "result": stage_output,
        })

        # Extract 'result' from stage output if it's a standard tool result dict
        if isinstance(stage_output, dict) and "result" in stage_output:
            if not stage_output.get("success", True):
                return {
                    "success": False,
                    "result": None,
                    "stage_results": stage_results,
                    "failed_stage": idx,
                    "error": (
                        f"Stage {idx} ({tool_name!r}) failed: "
                        f"{stage_output.get('error', 'unknown error')}"
                    ),
                }
            current_input = stage_output["result"]
        else:
            current_input = stage_output

    return {
        "success": True,
        "result": current_input,
        "stage_results": stage_results,
        "failed_stage": -1,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Registration function
# ---------------------------------------------------------------------------

def register_extended_tools(
    tool_manager: ToolManager,
    filesystem: Any = None,
) -> None:
    """
    Register all extended tools with the given ToolManager.

    Call this after ``register_builtin_tools()`` during kernel boot::

        from battousai.tools import register_builtin_tools
        from battousai.tools_extended import register_extended_tools

        register_builtin_tools(kernel.tools, kernel.filesystem)
        register_extended_tools(kernel.tools, kernel.filesystem)

    Args:
        tool_manager — The kernel's ToolManager instance.
        filesystem   — Optional VirtualFilesystem reference (currently unused
                       by extended tools but available for future expansion).
    """
    # Populate pipeline registry with all extended tool callables
    _PIPELINE_TOOL_REGISTRY.update({
        "http_client":     _http_client,
        "python_repl":     _python_repl,
        "json_processor":  _json_processor,
        "text_analyzer":   _text_analyzer,
        "vector_store":    _vector_store,
        "key_value_db":    _key_value_db,
        "task_queue":      _task_queue,
        "cron_scheduler":  _cron_scheduler,
    })

    tool_manager.register(ToolSpec(
        name="http_client",
        description=TOOL_SPEC_HTTP_CLIENT["description"],
        callable=_http_client,
        is_simulated=True,
        rate_limit=10,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="python_repl",
        description=TOOL_SPEC_PYTHON_REPL["description"],
        callable=_python_repl,
        is_simulated=False,
        rate_limit=20,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="json_processor",
        description=TOOL_SPEC_JSON_PROCESSOR["description"],
        callable=_json_processor,
        is_simulated=False,
        rate_limit=50,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="text_analyzer",
        description=TOOL_SPEC_TEXT_ANALYZER["description"],
        callable=_text_analyzer,
        is_simulated=False,
        rate_limit=30,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="vector_store",
        description=TOOL_SPEC_VECTOR_STORE["description"],
        callable=_vector_store,
        is_simulated=False,
        rate_limit=50,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="key_value_db",
        description=TOOL_SPEC_KEY_VALUE_DB["description"],
        callable=_key_value_db,
        is_simulated=False,
        rate_limit=100,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="task_queue",
        description=TOOL_SPEC_TASK_QUEUE["description"],
        callable=_task_queue,
        is_simulated=False,
        rate_limit=50,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="cron_scheduler",
        description=TOOL_SPEC_CRON_SCHEDULER["description"],
        callable=_cron_scheduler,
        is_simulated=False,
        rate_limit=20,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="data_pipeline",
        description=TOOL_SPEC_DATA_PIPELINE["description"],
        callable=_data_pipeline,
        is_simulated=False,
        rate_limit=10,
        rate_window=10,
    ))

# Extended Tools

The `tools_extended.py` module provides 9 additional tools beyond the built-in set. Register them after booting:

```python
from battousai.tools_extended import register_extended_tools

kernel.boot()
register_extended_tools(kernel.tools, kernel.filesystem)
```

---

## `http_client`

Simulated HTTP client supporting GET, POST, PUT, and DELETE.

**Rate limit:** 10 calls / 10 ticks | **Simulated:** Yes

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required | Target URL |
| `method` | `str` | `"GET"` | HTTP method: GET, POST, PUT, DELETE |
| `body` | `dict` | `None` | Request body (JSON-serialised) |
| `headers` | `dict` | `None` | Request headers dict |
| `timeout` | `int` | `30` | Timeout hint (unused in simulation) |

### Returns

```python
{
    "status": int,       # HTTP status code
    "body": Any,         # Response body
    "headers": dict,     # Response headers
    "latency_ms": int,   # Simulated latency
    "simulated": True,
}
```

### Example

```python
# GET request
r = self.use_tool("http_client", url="https://api.example.com/users", method="GET")
if r.ok:
    data = r.value
    self.log(f"Status: {data['status']}, Users: {data['body']['users']}")

# POST request with body
r = self.use_tool("http_client",
    url="https://httpbin.org/post",
    method="POST",
    body={"name": "Battousai Agent", "task": "research"},
)
if r.ok:
    self.log(f"POST response: {r.value['status']}")
```

Built-in mock URLs: `https://api.example.com/users`, `https://api.example.com/status`, `https://httpbin.org/get`, `https://httpbin.org/post`. Unknown URLs return a generic 200 response.

---

## `python_repl`

Safe Python expression evaluator with restricted builtins.

**Rate limit:** 20 calls / 10 ticks | **Real execution:** Yes (sandboxed)

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `code` | `str` | Python expression or statement to evaluate |

### Returns

```python
{
    "result": str,    # repr() of the evaluated result
    "success": bool,
    "error": str,     # error message if success=False
}
```

### Allowed builtins

`abs`, `round`, `min`, `max`, `sum`, `sorted`, `len`, `range`, `enumerate`, `zip`, `map`, `filter`, `int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, `set`, `isinstance`, `type`, `pi`, `e`, `sqrt`, `log`, `log2`, `log10`, `floor`, `ceil`, `pow`, `sin`, `cos`, `tan`, `factorial`

### Blocked patterns

`import`, `exec`, `eval`, `open`, `getattr`, `setattr`, `__import__`, `__builtins__`, `lambda`, `yield`, `async`, `await`

### Examples

```python
r = self.use_tool("python_repl", code="sum(range(1, 101))")
print(r.value["result"])  # "5050"

r = self.use_tool("python_repl", code="[x**2 for x in range(5)]")
print(r.value["result"])  # "[0, 1, 4, 9, 16]"

r = self.use_tool("python_repl", code="sorted([3,1,4,1,5,9,2,6])")
print(r.value["result"])  # "[1, 1, 2, 3, 4, 5, 6, 9]"

# Blocked
r = self.use_tool("python_repl", code="__import__('os').system('ls')")
print(r.value["success"])  # False
print(r.value["error"])    # "Blocked pattern detected: '__import__'"
```

---

## `json_processor`

Parse, query (dot-notation), and transform JSON data.

**Rate limit:** 50 calls / 10 ticks

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `operation` | `str` | One of: `parse`, `stringify`, `query`, `set`, `delete`, `keys`, `merge` |
| `data` | `Any` | Primary input (dict or JSON string) |
| `path` | `str` | Dot-notation path (e.g. `"user.address.city"`) for query/set/delete |
| `value` | `Any` | Value to assign (for `set`) |
| `data2` | `Any` | Second object (for `merge`) |
| `indent` | `int` | JSON indentation for `stringify` (default 2) |

### Examples

```python
# Parse JSON string
r = self.use_tool("json_processor",
    operation="parse",
    data='{"name": "Alice", "age": 30}',
)
obj = r.value["result"]  # {"name": "Alice", "age": 30}

# Query nested value
r = self.use_tool("json_processor",
    operation="query",
    data={"user": {"address": {"city": "Paris"}}},
    path="user.address.city",
)
print(r.value["result"])  # "Paris"

# Set a nested value
r = self.use_tool("json_processor",
    operation="set",
    data={"user": {"name": "Alice"}},
    path="user.email",
    value="alice@example.com",
)
# result: {"user": {"name": "Alice", "email": "alice@example.com"}}

# Deep merge
r = self.use_tool("json_processor",
    operation="merge",
    data={"a": 1, "b": {"x": 10}},
    data2={"b": {"y": 20}, "c": 3},
)
# result: {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
```

---

## `text_analyzer`

Analyse text for linguistic metrics including sentiment and readability.

**Rate limit:** 30 calls / 10 ticks

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Text to analyse |

### Returns

```python
{
    "word_count": int,
    "char_count": int,          # characters excluding spaces
    "sentence_count": int,
    "avg_word_length": float,
    "sentiment": str,           # "positive" | "negative" | "neutral"
    "sentiment_score": float,   # -1.0 to 1.0
    "positive_words": List[str],
    "negative_words": List[str],
    "readability": float,       # Flesch-Kincaid grade estimate
    "top_words": List[str],     # top 5 non-stop-words
}
```

### Example

```python
text = "Quantum computing represents an exciting breakthrough in technology."
r = self.use_tool("text_analyzer", text=text)
if r.ok:
    a = r.value
    self.log(f"Words: {a['word_count']}, Sentiment: {a['sentiment']}")
    self.log(f"Top words: {a['top_words']}")
    # Words: 9, Sentiment: positive
    # Top words: ['quantum', 'computing', 'represents', 'exciting', 'breakthrough']
```

---

## `vector_store`

In-memory vector similarity store using cosine similarity (pure Python — no NumPy).

**Rate limit:** 50 calls / 10 ticks

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `operation` | `str` | required | `add`, `search`, `delete`, `list`, `clear` |
| `collection` | `str` | `"default"` | Collection namespace |
| `id` | `str` | — | Vector identifier (add/delete) |
| `vector` | `List[float]` | — | Embedding vector (add/search) |
| `metadata` | `dict` | `{}` | Arbitrary metadata stored with the vector |
| `top_k` | `int` | `5` | Number of results to return (search) |

### Example

```python
# Add vectors
self.use_tool("vector_store",
    operation="add",
    collection="docs",
    id="doc_1",
    vector=[0.1, 0.9, 0.3, 0.7],
    metadata={"title": "Quantum Computing Overview"},
)

self.use_tool("vector_store",
    operation="add",
    collection="docs",
    id="doc_2",
    vector=[0.8, 0.2, 0.6, 0.1],
    metadata={"title": "Classical Computing Basics"},
)

# Search by similarity
r = self.use_tool("vector_store",
    operation="search",
    collection="docs",
    vector=[0.15, 0.85, 0.25, 0.65],
    top_k=2,
)
results = r.value["result"]
# [{"id": "doc_1", "score": 0.998, "metadata": {"title": "Quantum Computing Overview"}},
#  {"id": "doc_2", "score": 0.412, "metadata": {"title": "Classical Computing Basics"}}]

# List all IDs
r = self.use_tool("vector_store", operation="list", collection="docs")
print(r.value["result"])  # ["doc_1", "doc_2"]
```

Collections are isolated namespaces — `collection="docs"` and `collection="papers"` are independent.

---

## `key_value_db`

Persistent (in-process) key-value store with optional TTL per key.

**Rate limit:** 100 calls / 10 ticks

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `operation` | `str` | required | `get`, `set`, `delete`, `exists`, `keys`, `flush`, `ttl` |
| `db` | `str` | `"default"` | Database namespace |
| `key` | `str` | — | Key to operate on |
| `value` | `Any` | — | Value to store (set) |
| `ttl_ticks` | `int` | `None` | Expiry TTL in ticks (None = no expiry) |
| `current_tick` | `int` | `0` | Current system tick for TTL evaluation |

### TTL return values for `ttl` operation

| Return | Meaning |
|---|---|
| `-2` | Key does not exist |
| `-1` | Key has no expiry |
| `N ≥ 0` | Ticks remaining until expiry |

### Example

```python
# Set a value with TTL
self.use_tool("key_value_db",
    operation="set",
    db="cache",
    key="search_result",
    value={"data": "..."},
    ttl_ticks=20,
    current_tick=tick,
)

# Get a value
r = self.use_tool("key_value_db",
    operation="get",
    db="cache",
    key="search_result",
    current_tick=tick,
)
if r.value["result"] is not None:
    cached = r.value["result"]

# Check TTL remaining
r = self.use_tool("key_value_db",
    operation="ttl", db="cache", key="search_result", current_tick=tick
)
ttl_remaining = r.value["result"]  # e.g. 15

# List all non-expired keys
r = self.use_tool("key_value_db",
    operation="keys", db="cache", current_tick=tick
)
```

---

## `task_queue`

Priority task queue shared across agents. Lower priority integer = higher urgency (min-heap).

**Rate limit:** 50 calls / 10 ticks

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `operation` | `str` | required | `push`, `pop`, `peek`, `size`, `list`, `clear` |
| `queue` | `str` | `"default"` | Queue namespace |
| `task` | `dict` | — | Task dict to enqueue (push) |
| `priority` | `int` | `5` | Priority 0–9; lower = higher urgency |

### Example

```python
# Push tasks
self.use_tool("task_queue",
    operation="push",
    queue="jobs",
    task={"type": "urgent", "data": "process_now"},
    priority=1,
)
self.use_tool("task_queue",
    operation="push",
    queue="jobs",
    task={"type": "normal", "data": "process_later"},
    priority=5,
)

# Pop highest-priority task
r = self.use_tool("task_queue", operation="pop", queue="jobs")
item = r.value["result"]
# {"priority": 1, "task": {"type": "urgent", "data": "process_now"}}

# Peek without removing
r = self.use_tool("task_queue", operation="peek", queue="jobs")

# Queue size
r = self.use_tool("task_queue", operation="size", queue="jobs")
print(r.value["result"])  # 1
```

---

## `cron_scheduler`

Tick-based cron-like scheduler for recurring tool invocations.

**Rate limit:** 20 calls / 10 ticks

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `operation` | `str` | required | `register`, `unregister`, `tick`, `list` |
| `schedule` | `str` | `"default"` | Schedule namespace |
| `name` | `str` | — | Unique entry name (register/unregister) |
| `every_n_ticks` | `int` | — | Run interval in ticks |
| `tool` | `str` | — | Tool name to fire |
| `args` | `dict` | `{}` | Args for the tool on each fire |
| `current_tick` | `int` | `0` | Current system tick (tick operation) |

!!! note "Non-blocking"
    The cron scheduler does NOT directly call tools. The `tick` operation returns a list of **due entries** that your agent should dispatch. This keeps the tool stateless with respect to the kernel.

### Example

```python
# Register a recurring health check every 10 ticks
self.use_tool("cron_scheduler",
    operation="register",
    schedule="my_cron",
    name="health_check",
    every_n_ticks=10,
    tool="web_search",
    args={"query": "system status"},
    current_tick=tick,
)

# In think(), advance the clock and dispatch due entries
def think(self, tick: int) -> None:
    r = self.use_tool("cron_scheduler",
        operation="tick",
        schedule="my_cron",
        current_tick=tick,
    )
    for entry in r.value["result"]:
        self.log(f"Cron fired: {entry['name']} → {entry['tool']}")
        # Dispatch the tool
        self.use_tool(entry["tool"], **entry["args"])

    self.yield_cpu()
```

---

## `data_pipeline`

Chain multiple tool invocations sequentially. Each stage's output is injected into the next stage's args.

**Rate limit:** 10 calls / 10 ticks

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `stages` | `List[dict]` | Ordered list of stage configs |
| `initial_input` | `Any` | Initial data fed to the first stage |

Each stage dict:
```python
{
    "tool": "tool_name",
    "args": {},               # fixed args for this stage
    "input_key": "data",      # which arg key receives previous stage's result
                              # (None = don't inject; useful for first stage)
}
```

### Returns

```python
{
    "success": bool,
    "result": Any,                  # final stage output
    "stage_results": List[dict],    # per-stage results
    "failed_stage": int,            # index of first failure (-1 if all succeeded)
    "error": str,
}
```

### Example

```python
# Pipeline: parse JSON → query a field → analyze the text
r = self.use_tool("data_pipeline",
    stages=[
        {
            "tool": "json_processor",
            "args": {"operation": "parse", "data": '{"body": "Quantum computing is amazing!"}'},
            "input_key": None,       # no injection for first stage
        },
        {
            "tool": "json_processor",
            "args": {"operation": "query", "path": "body"},
            "input_key": "data",     # previous result → args["data"]
        },
        {
            "tool": "text_analyzer",
            "args": {},
            "input_key": "text",     # previous result → args["text"]
        },
    ],
)
if r.ok and r.value["success"]:
    final = r.value["result"]
    self.log(f"Pipeline result: sentiment={final['sentiment']}, words={final['word_count']}")
```

---

## Related Pages

- [Built-in Tools](builtin.md) — the 5 standard tools always available
- [Agent API](../agents/api.md) — `use_tool()` method reference
- [Custom Agents](../agents/custom.md) — examples using tools in workflows

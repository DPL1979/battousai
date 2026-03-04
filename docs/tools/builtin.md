# Built-in Tools

The `tools.py` module registers five built-in tools with the Battousai kernel during boot. These tools are always available to all agents (unless access control is applied).

---

## How Tools Work

Tools are Python callables registered with the `ToolManager`. Agents invoke them via the `access_tool` syscall:

```python
# From within an agent's think() method:
result = self.use_tool("tool_name", param1="value1", param2="value2")
if result.ok:
    output = result.value
else:
    self.log(f"Tool error: {result.error}")
```

The `ToolManager` handles:
- **Registration** — tools registered by name during boot
- **Access control** — per-agent allowlists (empty = open to all)
- **Rate limiting** — max calls per agent per time window
- **Usage logging** — every call recorded for audit and analytics

---

## `calculator`

Safe arithmetic expression evaluator.

**Rate limit:** 50 calls / 10 ticks

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `expression` | `str` | Arithmetic expression to evaluate |

### Supported operations

- Basic arithmetic: `+`, `-`, `*`, `/`, `**`, `%`, `//`
- Parentheses
- Math functions: `abs`, `round`, `min`, `max`, `sqrt`, `log`, `log2`, `log10`, `floor`, `ceil`, `pow`
- Constants: `pi`, `e`

### Returns

A string containing the result, or `"ERROR: ..."` if evaluation fails.

### Examples

```python
# Basic arithmetic
r = self.use_tool("calculator", expression="2 ** 10")
print(r.value)  # "1024"

r = self.use_tool("calculator", expression="(3.14159 * 5 ** 2)")
print(r.value)  # "78.53975"

# Math functions
r = self.use_tool("calculator", expression="sqrt(144)")
print(r.value)  # "12.0"

r = self.use_tool("calculator", expression="round(pi, 4)")
print(r.value)  # "3.1416"

# Error case
r = self.use_tool("calculator", expression="1 / 0")
print(r.value)  # "ERROR: division by zero"
```

!!! note "Security"
    The calculator uses `eval()` with a restricted namespace — only whitelisted math functions are available. No `import`, no file access, no arbitrary code execution.

---

## `web_search`

Simulated web search returning structured results.

**Rate limit:** 5 calls / 10 ticks  
**Simulated:** Yes — returns pre-seeded quantum computing research data

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | Search query string |

### Returns

```python
{
    "query": str,             # The query string
    "source": str,            # "SimulatedSearchEngine v1.0"
    "results": [
        {
            "title": str,     # "Search result for: {query}"
            "snippet": str,   # The result text
            "url": str,       # "sim://search/1"
        }
    ],
    "total_results": int,     # 1
    "simulated": True,
}
```

### Example

```python
r = self.use_tool("web_search", query="quantum computing basics")
if r.ok:
    data = r.value
    for result in data["results"]:
        self.log(f"Snippet: {result['snippet'][:100]}")
```

Built-in search topics with canned responses:
- `"quantum computing basics"`
- `"quantum computing applications"`
- `"quantum computing challenges"`
- `"quantum supremacy milestones"`
- Any other query returns the `"default"` response

!!! tip "Production use"
    In production, replace the `_simulated_web_search` function body with a real API call (Bing, Google, Brave, etc.) using the same return structure.

---

## `code_executor`

Simulated code execution environment.

**Rate limit:** 3 calls / 10 ticks  
**Simulated:** Yes — does not actually run code

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | `str` | required | Code to execute |
| `language` | `str` | `"python"` | Language identifier |

### Returns

```python
{
    "language": str,         # e.g. "python"
    "lines_executed": int,   # number of lines in code
    "stdout": str,           # "[SIMULATED] Executed N line(s) of python code successfully."
    "stderr": str,           # ""
    "exit_code": int,        # 0
    "simulated": True,
}
```

### Example

```python
code = """
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(result)
"""

r = self.use_tool("code_executor", code=code, language="python")
if r.ok:
    self.log(f"Exit code: {r.value['exit_code']}")
    self.log(f"Output: {r.value['stdout']}")
```

!!! tip "Python REPL for real execution"
    For actual Python expression evaluation, use the `python_repl` extended tool, which executes code using a restricted `eval()` sandbox. See [Extended Tools](extended.md).

---

## `file_reader`

Read a file from the Battousai virtual filesystem by path.

**Rate limit:** 20 calls / 10 ticks

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | Absolute filesystem path |
| `agent_id` | `str` | `"kernel"` | Agent performing the read (for permission checks) |

### Returns

The file's contents (whatever Python object was stored — string, dict, list, etc.).

### Example

```python
r = self.use_tool("file_reader", path="/shared/results/summary.txt")
if r.ok:
    content = r.value
    self.log(f"File contents ({len(str(content))} chars): {str(content)[:80]}")
else:
    self.log(f"Read failed: {r.error}")
```

!!! note "Prefer the syscall wrapper"
    From within an agent, prefer `self.read_file(path)` over `self.use_tool("file_reader", ...)`. The syscall wrapper passes the correct `agent_id` for permission enforcement.

---

## `file_writer`

Write data to a file in the Battousai virtual filesystem.

**Rate limit:** 20 calls / 10 ticks

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | Absolute filesystem path |
| `data` | `Any` | required | Data to write |
| `agent_id` | `str` | `"kernel"` | Agent performing the write |

### Returns

A string `"OK: wrote N chars to '/path/to/file'"` on success.

### Example

```python
r = self.use_tool("file_writer", path="/shared/data.txt", data="Hello, Battousai!")
if r.ok:
    self.log(r.value)  # "OK: wrote 12 chars to '/shared/data.txt'"
```

!!! note "Prefer the syscall wrapper"
    From within an agent, prefer `self.write_file(path, data)` which handles the `agent_id` automatically.

---

## `ToolManager` Reference

For kernel-level code or custom tool registration:

```python
from battousai.tools import ToolManager, ToolSpec

# Register a custom tool
def my_tool(param: str) -> dict:
    return {"result": param.upper(), "length": len(param)}

kernel.tools.register(ToolSpec(
    name="my_tool",
    description="Converts input to uppercase and returns its length.",
    callable=my_tool,
    rate_limit=10,
    rate_window=10,
    is_simulated=False,
))

# List registered tools
print(kernel.tools.list_tools())
# ['calculator', 'code_executor', 'file_reader', 'file_writer', 'my_tool', 'web_search']

# Get tool stats
stats = kernel.tools.stats()
# {
#   "total_calls": 15,
#   "calls_by_tool": {"web_search": 5, "calculator": 10},
#   "calls_by_agent": {"worker_0003": 8, "worker_0004": 7},
#   "registered_tools": [...],
# }

# Grant/revoke per-agent access
kernel.tools.grant_access("my_tool", "worker_0003")   # only worker_0003 can use
kernel.tools.revoke_access("my_tool", "worker_0003")  # remove access

# View usage log
for record in kernel.tools.usage_log():
    print(f"{record.tool_name} by {record.agent_id} at tick {record.tick}")
```

---

## `ToolSpec` Dataclass

```python
@dataclass
class ToolSpec:
    name: str                          # Unique tool identifier
    description: str                   # Human/agent-readable description
    callable: Callable[..., Any]       # The Python function implementing the tool
    allowed_agents: Set[str] = ...     # Empty = open to all agents
    rate_limit: int = 0                # Max calls per agent per window (0 = unlimited)
    rate_window: int = 10              # Tick window for rate limiting
    is_simulated: bool = False         # True for tools returning synthetic data
```

---

## Related Pages

- [Extended Tools](extended.md) — 9 additional tools for complex workflows
- [Agent API](../agents/api.md) — `use_tool()` method reference
- [Kernel](../architecture/kernel.md) — `access_tool` syscall dispatch

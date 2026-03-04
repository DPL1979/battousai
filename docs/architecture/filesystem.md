# Virtual Filesystem

The `filesystem.py` module implements a hierarchical in-memory filesystem for Battousai. Agents write findings, logs, and inter-agent data through this layer.

---

## Directory Layout

The kernel creates a standard directory tree during `boot()`:

```
/
├── agents/
│   └── {agent_id}/
│       └── workspace/       ← Each agent's private working directory
├── shared/
│   └── results/             ← Cross-agent output publication
└── system/
    └── logs/                ← OS log files (written by Logger)
```

- `/agents/{agent_id}/` — created automatically when an agent is spawned
- `/agents/{agent_id}/workspace/` — private scratch space for each agent
- `/shared/` — cross-agent collaboration area; world-readable by default
- `/shared/results/` — canonical location for published summaries and outputs
- `/system/logs/` — structured log files written by the logger per tick

---

## Permissions

Each file has three permission flags:

| Flag | Description | Default |
|---|---|---|
| `owner_read` | Owner agent can read | `True` |
| `owner_write` | Owner agent can write | `True` |
| `world_read` | All agents can read | `True` |
| `world_write` | All agents can write | `False` |

The `kernel` agent_id **always bypasses** all permission checks.

`FileMetadata.can_read(agent_id)` and `can_write(agent_id)` implement the logic:

```python
def can_read(self, agent_id: str) -> bool:
    if agent_id == "kernel": return True
    if agent_id == self.created_by: return self.owner_read
    if agent_id in self.group_agents: return True
    return self.world_read
```

---

## `VirtualFilesystem` API

```python
class VirtualFilesystem:
    # Write
    def write_file(
        self,
        agent_id: str,
        path: str,
        data: Any,
        create_parents: bool = True,
        overwrite: bool = True,
        world_readable: bool = True,
        world_writable: bool = False,
    ) -> FSFile

    # Read
    def read_file(self, agent_id: str, path: str) -> Any

    # Delete
    def delete_file(self, agent_id: str, path: str) -> bool

    # Directory operations
    def list_dir(self, agent_id: str, path: str) -> List[str]
    def mkdir(self, path: str) -> FSDirectory
    def exists(self, path: str) -> bool

    # Metadata
    def get_metadata(self, path: str) -> Optional[FileMetadata]

    # Debug
    def tree(self, path: str = "/", indent: int = 0) -> str
    def stats(self) -> Dict[str, Any]
```

---

## Writing Files

From within an agent:

```python
# Write a string to a private file
self.write_file(
    f"/agents/{self.agent_id}/workspace/notes.txt",
    "Important finding: ..."
)

# Write a dict (stored as-is; any Python object can be stored)
self.write_file("/shared/results/data.json", {"key": "value", "count": 42})

# Write with restricted access
self.syscall(
    "write_file",
    path="/agents/self/private.txt",
    data="secret",
    world_readable=False,
)
```

`create_parents=True` (the default) means you don't need to pre-create directories — they are created automatically.

---

## Reading Files

```python
result = self.read_file("/shared/results/summary.txt")
if result.ok:
    content = result.value
    self.log(f"Summary length: {len(str(content))}")
else:
    self.log(f"Read failed: {result.error}")
```

Common errors:
- `FileNotFoundError` — path does not exist
- `PermissionError` — agent lacks read access to that file
- `NotADirectoryError` — tried to read a directory as a file

---

## Listing Directories

```python
result = self.syscall("list_dir", path="/shared/results")
if result.ok:
    files = result.value  # ["summary.txt", "data.json"]
    for filename in files:
        self.log(f"Found: {filename}")
```

---

## File Metadata

Files track metadata automatically:

```python
@dataclass
class FileMetadata:
    created_by: str      # agent_id that created the file
    created_at: int      # tick of creation
    modified_at: int     # tick of last write
    size: int            # approximate character/byte count
    owner_read: bool
    owner_write: bool
    world_read: bool
    world_write: bool
    group_agents: Set[str]   # agents with group access
    wall_created: float      # wall-clock timestamp
```

Access metadata directly via the filesystem (for external/kernel code):

```python
meta = kernel.filesystem.get_metadata("/shared/results/summary.txt")
if meta:
    print(f"Created by: {meta.created_by}")
    print(f"Size: {meta.size} bytes")
    print(f"Modified at tick: {meta.modified_at}")
```

---

## Error Types

```python
class FSError(Exception): ...
class FileNotFoundError(FSError): ...    # path does not exist
class PermissionError(FSError): ...      # lacks access
class NotADirectoryError(FSError): ...   # expected dir but got file (or vice versa)
class FileExistsError(FSError): ...      # file exists and overwrite=False
```

These are raised internally and caught by the kernel's `_syscall_write_file` / `_syscall_read_file` handlers, which return `SyscallResult(ok=False, error=str(exc))`.

---

## Tree Rendering

For debugging, print the full filesystem tree:

```python
print(kernel.filesystem.tree("/"))
```

Example output:
```
//
  agents/
    coordinator_0002/
      workspace/
        plan.txt (245b)
    worker_0003/
      workspace/
        results.json (892b)
  shared/
    results/
      summary.txt (3936b)
  system/
    logs/
```

---

## Statistics

```python
stats = kernel.filesystem.stats()
# {
#   "total_files": 17,
#   "total_size_bytes": 8234,
#   "files_by_agent": {
#       "coordinator_0002": 2,
#       "worker_0003": 3,
#       "worker_0004": 2,
#       "kernel": 10,
#   }
# }
```

---

## Common Patterns

### Agent-Private Workspace

```python
class DataAgent(Agent):
    def think(self, tick: int) -> None:
        # Write to own workspace (always allowed)
        self.write_file(
            f"/agents/{self.agent_id}/workspace/output_{tick}.txt",
            f"Data from tick {tick}"
        )
        self.yield_cpu()
```

### Shared Results Pattern

```python
class CoordinatorAgent(Agent):
    def _publish_summary(self, summary: str) -> None:
        result = self.write_file("/shared/results/summary.txt", summary)
        if result.ok:
            self.log("Summary published to /shared/results/summary.txt")

class ReaderAgent(Agent):
    def think(self, tick: int) -> None:
        r = self.read_file("/shared/results/summary.txt")
        if r.ok:
            self.log(f"Found summary ({len(str(r.value))} chars)")
        self.yield_cpu()
```

### Listing and Processing Files

```python
class IndexAgent(Agent):
    def think(self, tick: int) -> None:
        r = self.syscall("list_dir", path="/shared/results")
        if r.ok:
            for filename in r.value:
                content = self.read_file(f"/shared/results/{filename}")
                if content.ok:
                    self.log(f"{filename}: {len(str(content.value))} chars")
        self.yield_cpu()
```

---

## Related Pages

- [Kernel](kernel.md) — `write_file`, `read_file`, `list_dir` syscall handlers
- [Agent API](../agents/api.md) — `write_file`, `read_file` convenience wrappers
- [Architecture Overview](overview.md) — filesystem's role in the layered stack

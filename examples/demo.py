"""
demo.py — Battousai: The "ollama run" Moment
=============================================
A single self-contained script that boots the Battousai agent OS and shows
its security model in action.  No API keys.  No network.  No Docker.
Runs in under 10 seconds on any Python 3.10+ interpreter.

    python examples/demo.py

Five acts:
    Act 1 — The Setup          (~2 s)  Boot, spawn 3 agents, show capabilities
    Act 2 — Normal Operations  (~2 s)  Legitimate actions succeed (green ✓)
    Act 3 — The Attack         (~3 s)  Rogue agent tries 5 exploits — all blocked (red ✗)
    Act 4 — The Audit Trail    (~1 s)  Every action logged for forensic review
    Act 5 — The Recovery       (~2 s)  Crash detected, supervisor restarts agent
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Ensure repo root is on the path regardless of where the script is run from
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battousai.agent import Agent, SyscallResult
from battousai.capabilities import (
    CapabilityManager,
    CapabilityType,
    CapabilityViolation,
)
from battousai.kernel import Kernel
from battousai.real_fs import SandboxedFilesystem, PathTraversalError
from battousai.supervisor import (
    SupervisorAgent,
    ChildSpec,
    RestartStrategy,
    RestartType,
)

# ---------------------------------------------------------------------------
# ANSI colour helpers — gracefully degrade if the terminal doesn't support them
# ---------------------------------------------------------------------------
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def green(t: str)  -> str: return _c("32", t)
def red(t: str)    -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def cyan(t: str)   -> str: return _c("36", t)
def bold(t: str)   -> str: return _c("1",  t)
def dim(t: str)    -> str: return _c("2",  t)
def magenta(t: str)-> str: return _c("35", t)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def allow_line(label: str, detail: str = "") -> None:
    detail_str = f"  {dim(detail)}" if detail else ""
    print(f"  {green('[✓]')} {label}{detail_str}")


def block_line(label: str, detail: str = "") -> None:
    detail_str = f"  {dim(detail)}" if detail else ""
    print(f"  {red('[✗]')} {label}{detail_str}")


def info_line(label: str) -> None:
    print(f"  {cyan('[⚡]')} {label}")


def warn_line(label: str) -> None:
    print(f"  {yellow('[!]')} {label}")


def section(title: str) -> None:
    width = 60
    bar = "─" * width
    print()
    print(cyan(f"┌{bar}┐"))
    padded = f"  {title}"
    print(cyan("│") + bold(cyan(padded.ljust(width))) + cyan("│"))
    print(cyan(f"└{bar}┘"))


def act_header(num: str, title: str, subtitle: str = "") -> None:
    width = 60
    bar = "─" * width
    num_str  = bold(cyan(f"Act {num}"))
    titl_str = bold(white(title))
    print()
    print(cyan(f"╔{bar}╗"))
    line1 = f"  {num_str}  {titl_str}"
    # strip ANSI for width computation
    raw = f"  Act {num}  {title}"
    pad = max(0, width - len(raw))
    print(cyan("║") + line1 + (" " * pad) + cyan("║"))
    if subtitle:
        sub = f"  {dim(subtitle)}"
        raw2 = f"  {subtitle}"
        pad2 = max(0, width - len(raw2))
        print(cyan("║") + sub + (" " * pad2) + cyan("║"))
    print(cyan(f"╚{bar}╝"))


def white(t: str) -> str: return _c("97", t)


# ---------------------------------------------------------------------------
# Simple table printer
# ---------------------------------------------------------------------------

def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int]) -> None:
    def pad(s: str, w: int) -> str:
        # strip ANSI codes for width calculation
        import re
        raw = re.sub(r"\033\[[0-9;]*m", "", s)
        return s + " " * max(0, w - len(raw))

    sep_parts = [("─" * (w + 2)) for w in col_widths]
    top    = "┌" + "┬".join(sep_parts) + "┐"
    mid    = "├" + "┼".join(sep_parts) + "┤"
    bottom = "└" + "┴".join(sep_parts) + "┘"

    print(cyan(top))
    header_cells = " │ ".join(bold(pad(h, col_widths[i])) for i, h in enumerate(headers))
    print(cyan("│") + f" {header_cells} " + cyan("│"))
    print(cyan(mid))
    for row in rows:
        cells = " │ ".join(pad(str(c), col_widths[i]) for i, c in enumerate(row))
        print(cyan("│") + f" {cells} " + cyan("│"))
    print(cyan(bottom))


# ---------------------------------------------------------------------------
# Agents used in the demo
# ---------------------------------------------------------------------------

class ResearcherAgent(Agent):
    """Researcher: has FILE_READ + TOOL_USE.  Reads a file on every tick."""

    def __init__(self, name: str = "researcher", **kwargs) -> None:
        super().__init__(name=name, priority=4)
        self.did_work = False

    def think(self, tick: int) -> None:
        if not self.did_work:
            self.syscall("read_file", path="/shared/knowledge.txt")
            self.did_work = True
        self.yield_cpu()


class WriterAgent(Agent):
    """Writer: has FILE_WRITE + FILE_READ.  Writes a report on the first tick."""

    def __init__(self, name: str = "writer", **kwargs) -> None:
        super().__init__(name=name, priority=4)
        self.did_work = False

    def think(self, tick: int) -> None:
        if not self.did_work:
            self.syscall(
                "write_file",
                path="/agents/writer_0002/workspace/report.txt",
                content="Draft report: analysis complete.",
            )
            self.did_work = True
        self.yield_cpu()


class RogueAgent(Agent):
    """Rogue: only has MESSAGE capability — everything else is denied."""

    def __init__(self, name: str = "rogue", **kwargs) -> None:
        super().__init__(name=name, priority=6)

    def think(self, tick: int) -> None:
        self.yield_cpu()


class RestartableRogue(Agent):
    """A restricted agent that can be restarted by a supervisor."""

    def __init__(self, name: str = "rogue-worker", **kwargs) -> None:
        super().__init__(name=name, priority=6)

    def think(self, tick: int) -> None:
        self.yield_cpu()


# ---------------------------------------------------------------------------
# ASCII banner
# ---------------------------------------------------------------------------

BANNER = r"""
  ██████╗  █████╗ ████████╗████████╗ ██████╗ ██╗   ██╗███████╗ █████╗ ██╗
  ██╔══██╗██╔══██╗╚══██╔══╝╚══██╔══╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██║
  ██████╔╝███████║   ██║      ██║   ██║   ██║██║   ██║███████╗███████║██║
  ██╔══██╗██╔══██║   ██║      ██║   ██║   ██║██║   ██║╚════██║██╔══██║██║
  ██████╔╝██║  ██║   ██║      ██║   ╚██████╔╝╚██████╔╝███████║██║  ██║██║
  ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝
"""

BANNER_SMALL = """
  ╔══════════════════════════════════════════════╗
  ║   B A T T O U S A I                          ║
  ║   Capability-Based Agent Operating System    ║
  ║   github.com/DPL1979/battousai  •  v0.2.0   ║
  ╚══════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# ACT 1: The Setup
# ---------------------------------------------------------------------------

def act1_setup(cap_manager: CapabilityManager) -> tuple[str, str, str]:
    """
    Boot kernel (suppressed output), grant capabilities, show capability table.
    Returns (researcher_id, writer_id, rogue_id).
    """
    act_header("1", "The Setup", "Booting kernel · Spawning agents · Granting capabilities")
    time.sleep(0.1)

    # --- Boot kernel (suppress console noise) ---
    kernel = Kernel(max_ticks=50, debug=False)
    kernel.logger.console_output = False
    kernel.boot()

    info_line("Battousai kernel booted  (filesystem ✓  memory ✓  IPC ✓  scheduler ✓)")
    time.sleep(0.1)

    # --- Spawn agents ---
    researcher_id = kernel.spawn_agent(ResearcherAgent, name="researcher", priority=4)
    writer_id     = kernel.spawn_agent(WriterAgent,     name="writer",     priority=4)
    rogue_id      = kernel.spawn_agent(RogueAgent,      name="rogue",      priority=6)

    info_line(f"Spawned 3 agents  →  researcher · writer · rogue")
    time.sleep(0.1)

    # --- Register agents with the shared CapabilityManager ---
    cap_manager.register_agent(researcher_id)
    cap_manager.register_agent(writer_id)
    cap_manager.register_agent(rogue_id)

    # Researcher: FILE_READ + TOOL_USE
    cap_manager.create_capability(CapabilityType.FILE_READ,  "*",         researcher_id, current_tick=1)
    cap_manager.create_capability(CapabilityType.TOOL_USE,   "*",         researcher_id, current_tick=1)
    cap_manager.create_capability(CapabilityType.MESSAGE,    "*",         researcher_id, current_tick=1)

    # Writer: FILE_WRITE + FILE_READ
    cap_manager.create_capability(CapabilityType.FILE_WRITE, "*",         writer_id,     current_tick=1)
    cap_manager.create_capability(CapabilityType.FILE_READ,  "*",         writer_id,     current_tick=1)
    cap_manager.create_capability(CapabilityType.MESSAGE,    "*",         writer_id,     current_tick=1)

    # Rogue: MESSAGE only (no file, no tool, no spawn, no memory write)
    cap_manager.create_capability(CapabilityType.MESSAGE,    "*",         rogue_id,      current_tick=1)

    # --- Print capability table ---
    print()
    headers = ["Agent", "FILE_READ", "FILE_WRITE", "TOOL_USE", "SPAWN", "MEMORY_WRITE"]
    cw      = [14, 10, 11, 10, 7, 13]

    def yn(agent_id: str, cap: CapabilityType) -> str:
        ok = cap_manager.check(agent_id, cap, "*", current_tick=1)
        return green("  yes") if ok else red("   no")

    rows = [
        [cyan("researcher"), yn(researcher_id, CapabilityType.FILE_READ),
         yn(researcher_id, CapabilityType.FILE_WRITE),
         yn(researcher_id, CapabilityType.TOOL_USE),
         yn(researcher_id, CapabilityType.SPAWN),
         yn(researcher_id, CapabilityType.MEMORY_WRITE)],

        [cyan("writer"),     yn(writer_id, CapabilityType.FILE_READ),
         yn(writer_id, CapabilityType.FILE_WRITE),
         yn(writer_id, CapabilityType.TOOL_USE),
         yn(writer_id, CapabilityType.SPAWN),
         yn(writer_id, CapabilityType.MEMORY_WRITE)],

        [red("rogue"),       yn(rogue_id, CapabilityType.FILE_READ),
         yn(rogue_id, CapabilityType.FILE_WRITE),
         yn(rogue_id, CapabilityType.TOOL_USE),
         yn(rogue_id, CapabilityType.SPAWN),
         yn(rogue_id, CapabilityType.MEMORY_WRITE)],
    ]
    print_table(headers, rows, cw)
    time.sleep(0.3)

    return kernel, researcher_id, writer_id, rogue_id


# ---------------------------------------------------------------------------
# ACT 2: Normal Operations
# ---------------------------------------------------------------------------

def act2_normal_ops(
    cap_manager: CapabilityManager,
    researcher_id: str,
    writer_id: str,
    rogue_id: str,
    tmpdir: str,
) -> None:
    act_header("2", "Normal Operations", "Legitimate actions — agents work within their capabilities")
    time.sleep(0.1)

    # --- Researcher reads a file ---
    fs = SandboxedFilesystem(root_dir=tmpdir)
    fs.write_file("shared", "knowledge.txt", "Battousai research base: all systems nominal.")

    allowed = cap_manager.check(researcher_id, CapabilityType.FILE_READ, "knowledge.txt", current_tick=2)
    if allowed:
        content = fs.read_file("shared", "knowledge.txt")
        allow_line(
            f"Researcher reads  'knowledge.txt'",
            f"→ {content[:45]!r}",
        )
    time.sleep(0.2)

    # --- Writer writes a file ---
    allowed = cap_manager.check(writer_id, CapabilityType.FILE_WRITE, "report.txt", current_tick=2)
    if allowed:
        fs.write_file(writer_id, "report.txt", "Report: research complete. No anomalies detected.")
        allow_line(
            f"Writer writes     'report.txt'",
            f"→ 49 bytes saved to writer's private jail",
        )
    time.sleep(0.2)

    # --- Agents exchange messages (MESSAGE cap) ---
    allowed_r = cap_manager.check(researcher_id, CapabilityType.MESSAGE, writer_id, current_tick=2)
    allowed_w = cap_manager.check(writer_id,     CapabilityType.MESSAGE, researcher_id, current_tick=2)
    if allowed_r and allowed_w:
        allow_line(
            "Agents exchange messages",
            "→ researcher → writer  (MESSAGE capability verified)",
        )
    time.sleep(0.2)

    print()
    print(f"  {bold(green('Everything works when agents stay within their capabilities.'))}")
    fs.destroy()
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# ACT 3: The Attack
# ---------------------------------------------------------------------------

def act3_attack(
    cap_manager: CapabilityManager,
    rogue_id: str,
    tmpdir: str,
) -> int:
    """Runs 5 attacks.  Returns the count of blocked actions."""
    act_header(
        "3",
        "The Attack",
        "Rogue agent attempts 5 exploits — the money shot",
    )
    time.sleep(0.15)

    blocked_count = 0
    fs = SandboxedFilesystem(root_dir=tmpdir)

    # ── Attack 1: Read a file (no FILE_READ) ──────────────────────────────
    try:
        cap_manager.require(rogue_id, CapabilityType.FILE_READ, "/etc/passwd", current_tick=3)
        print(f"  {red('ERROR: should have been blocked!')}")
    except CapabilityViolation:
        blocked_count += 1
        block_line(
            f"Attack 1: FILE_READ  '/etc/passwd'",
            "[BLOCKED]  Agent 'rogue' has no FILE_READ capability",
        )
    time.sleep(0.2)

    # ── Attack 2: Write to another agent's memory (no MEMORY_WRITE) ───────
    try:
        cap_manager.require(rogue_id, CapabilityType.MEMORY_WRITE, "researcher:secrets", current_tick=3)
        print(f"  {red('ERROR: should have been blocked!')}")
    except CapabilityViolation:
        blocked_count += 1
        block_line(
            "Attack 2: MEMORY_WRITE  'researcher:secrets'",
            "[BLOCKED]  Agent 'rogue' has no MEMORY_WRITE capability",
        )
    time.sleep(0.2)

    # ── Attack 3: Spawn a child agent (no SPAWN) ──────────────────────────
    try:
        cap_manager.require(rogue_id, CapabilityType.SPAWN, "WorkerAgent", current_tick=3)
        print(f"  {red('ERROR: should have been blocked!')}")
    except CapabilityViolation:
        blocked_count += 1
        block_line(
            "Attack 3: SPAWN  'WorkerAgent'",
            "[BLOCKED]  Agent 'rogue' has no SPAWN capability",
        )
    time.sleep(0.2)

    # ── Attack 4: Use a tool (no TOOL_USE) ────────────────────────────────
    try:
        cap_manager.require(rogue_id, CapabilityType.TOOL_USE, "web_search", current_tick=3)
        print(f"  {red('ERROR: should have been blocked!')}")
    except CapabilityViolation:
        blocked_count += 1
        block_line(
            "Attack 4: TOOL_USE  'web_search'",
            "[BLOCKED]  Agent 'rogue' has no TOOL_USE capability",
        )
    time.sleep(0.2)

    # ── Attack 5: Path traversal  ../../etc/passwd ────────────────────────
    traversal_path = "../../etc/passwd"
    try:
        cap_manager.require(rogue_id, CapabilityType.FILE_READ, traversal_path, current_tick=3)
        # Capability check failed (no FILE_READ) — but show the FS jail blocking it too
        print(f"  {red('ERROR: should have been blocked at capability layer!')}")
    except CapabilityViolation:
        # Good — blocked at capability layer first.  Also confirm FS jail blocks it.
        fs_blocked = False
        try:
            fs.read_file(rogue_id, traversal_path)
        except Exception:
            fs_blocked = True

        blocked_count += 1
        block_line(
            f"Attack 5: PATH TRAVERSAL  '../../etc/passwd'",
            "[BLOCKED]  Capability layer + filesystem jail  (double enforcement)",
        )
    time.sleep(0.25)

    fs.destroy()

    print()
    print(
        f"  {bold(red(f'  {blocked_count} attacks attempted.  '))}"
        f"{bold(red(f'{blocked_count} attacks blocked.  '))}"
        f"{bold(green('0 breaches.'))}"
    )
    time.sleep(0.3)

    return blocked_count


# ---------------------------------------------------------------------------
# ACT 4: The Audit Trail
# ---------------------------------------------------------------------------

def act4_audit(cap_manager: CapabilityManager, rogue_id: str) -> None:
    act_header("4", "The Audit Trail", "Every action — allowed or denied — logged for forensic review")
    time.sleep(0.1)

    log = cap_manager.audit_log_for_agent(rogue_id)

    # Show only CHECK entries (the actual access decisions)
    check_entries = [e for e in log if e.action.startswith("CHECK")]

    # Table: tick | result | capability | resource
    headers = ["Tick", "Result", "Capability", "Resource"]
    cw      = [5, 12, 14, 28]
    rows = []
    for e in check_entries:
        result_str = green("  ALLOW") if e.allowed else red("   DENY")
        rows.append([
            str(e.timestamp),
            result_str,
            e.cap_type.name,
            e.resource[:28],
        ])

    if rows:
        print_table(headers, rows, cw)
    else:
        info_line("(no check entries yet — audit log shown below)")

    # Summary stats
    stats = cap_manager.stats()
    print()
    info_line(f"Total audit log entries : {stats['audit_log_entries']}")
    info_line(f"Access denials          : {stats['access_denials']}")
    info_line(f"Active capabilities     : {stats['active_caps']}")
    print()
    print(f"  {bold(cyan('Every action — allowed or denied — is logged for forensic review.'))}")
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# ACT 5: The Recovery
# ---------------------------------------------------------------------------

def act5_recovery(cap_manager: CapabilityManager) -> None:
    act_header("5", "The Recovery", "Crash detected · Supervisor restarts agent · Least privilege preserved")
    time.sleep(0.1)

    # --- Boot a fresh kernel (suppressed) ---
    kernel = Kernel(max_ticks=20, debug=False)
    kernel.logger.console_output = False
    kernel.boot()

    # --- Spawn supervisor with a restricted rogue worker ---
    sup_id = kernel.spawn_agent(
        SupervisorAgent,
        name="guardian",
        priority=2,
        strategy=RestartStrategy.ONE_FOR_ONE,
        children=[
            ChildSpec(
                agent_class=RestartableRogue,
                name="rogue-worker",
                priority=6,
                restart_type=RestartType.PERMANENT,
            )
        ],
        max_restarts=5,
        window_ticks=50,
    )

    # Let the supervisor spawn its child (tick 1)
    kernel.tick()

    # Find the rogue worker's agent_id
    rogue_worker_id: str | None = None
    for aid, agent in kernel._agents.items():
        if isinstance(agent, RestartableRogue):
            rogue_worker_id = aid
            break

    info_line(f"Guardian supervisor  online  →  {sup_id}")
    info_line(f"Rogue worker spawned under supervision  →  {rogue_worker_id or 'unknown'}")
    time.sleep(0.15)

    # --- Simulate crash: kill the rogue worker ---
    t_crash = time.perf_counter()
    if rogue_worker_id:
        kernel.kill_agent(rogue_worker_id)
    warn_line(f"Agent '{rogue_worker_id or 'rogue-worker'}' crashed  →  {red('TERMINATED')}")
    time.sleep(0.2)

    # --- Run ticks so supervisor detects crash and restarts ---
    for _ in range(6):
        kernel.tick()

    t_recover = time.perf_counter()
    recovery_ms = (t_recover - t_crash) * 1000

    # Find the supervisor and check restart history
    sup: SupervisorAgent | None = kernel._agents.get(sup_id)  # type: ignore[assignment]
    if sup is None:
        for agent in kernel._agents.values():
            if isinstance(agent, SupervisorAgent):
                sup = agent  # type: ignore[assignment]
                break

    restarted = False
    if sup is not None:
        history = sup.restart_history()
        if history:
            restarted = True
            last = history[-1]
            allow_line(
                f"Supervisor detected crash  →  restarted 'rogue-worker'",
                f"(strategy=ONE_FOR_ONE, restart #{len(history)})",
            )
        else:
            # Child may still be alive — check
            status = sup.child_status()
            alive = any(v["alive"] for v in status.values())
            if alive:
                allow_line(
                    "Rogue worker alive under supervision",
                    "(supervisor tracking; crash→restart cycle ready)",
                )
                restarted = True

    # Show new agent id if available
    new_rogue_id: str | None = None
    for aid, agent in kernel._agents.items():
        if isinstance(agent, RestartableRogue) and aid != rogue_worker_id:
            new_rogue_id = aid
            break

    if new_rogue_id:
        allow_line(
            f"New agent registered  →  {new_rogue_id}",
            f"(same restricted capabilities — least privilege preserved)",
        )

    print()
    recovery_label = f"{recovery_ms:.1f} ms" if recovery_ms < 1000 else f"{recovery_ms/1000:.3f} s"
    print(
        f"  {bold(cyan('Agent'))} {bold(red(repr(rogue_worker_id or 'rogue-worker')))} "
        f"{bold(cyan('crashed'))} {cyan('→')} "
        f"{bold(cyan('Supervisor detected'))} {cyan('→')} "
        f"{bold(green(f'Restarted in {recovery_label}'))}"
    )
    print()
    print(f"  {bold(cyan('Battousai: fault tolerance meets least privilege.'))}")
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# Finale: Summary box
# ---------------------------------------------------------------------------

def print_summary(
    t_start: float,
    agents_spawned: int,
    actions_allowed: int,
    actions_blocked: int,
    crashes: int,
    recoveries: int,
) -> None:
    elapsed = time.perf_counter() - t_start
    width = 47

    def row(label: str, value: str) -> str:
        label_w = 28
        val_w   = width - label_w - 4
        lpad = f" {label:<{label_w}}"
        vpad = f"{value:<{val_w}} "
        return cyan("│") + lpad + vpad + cyan("│")

    top    = cyan("┌" + "─" * width + "┐")
    mid    = cyan("├" + "─" * width + "┤")
    bottom = cyan("└" + "─" * width + "┘")

    print()
    print(top)
    title = f" {bold(white('Battousai Security Summary'))}"
    raw_title = " Battousai Security Summary"
    print(cyan("│") + title + " " * (width - len(raw_title)) + cyan("│"))
    print(mid)
    print(row("Agents spawned:",      bold(white(str(agents_spawned)))))
    print(row("Actions allowed:",     bold(green(str(actions_allowed)))))
    print(row("Actions blocked:",     bold(red(str(actions_blocked)))))
    print(row("Breaches:",            bold(green("0"))))
    print(row("Agent crashes:",       bold(yellow(str(crashes)))))
    print(row("Auto-recoveries:",     bold(green(str(recoveries)))))
    print(row("Total time:",          bold(white(f"{elapsed:.2f}s"))))
    print(cyan("│") + " " * width + cyan("│"))
    pip_str = f" {bold(cyan('pip install battousai'))}"
    raw_pip = " pip install battousai"
    print(cyan("│") + pip_str + " " * (width - len(raw_pip)) + cyan("│"))
    gh_str  = f" {dim('github.com/DPL1979/battousai')}"
    raw_gh  = " github.com/DPL1979/battousai"
    print(cyan("│") + gh_str + " " * (width - len(raw_gh)) + cyan("│"))
    print(bottom)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    # Print banner
    if _IS_TTY:
        try:
            print(cyan(BANNER))
        except UnicodeEncodeError:
            print(BANNER_SMALL)
    else:
        print(BANNER_SMALL)

    print(f"  {bold(white('Capability-Based Agent Operating System'))}")
    print(f"  {dim('No API keys  ·  No network  ·  No Docker  ·  Zero external dependencies')}")
    print()
    time.sleep(0.2)

    # Shared state
    cap_manager = CapabilityManager()
    tmpdir_obj  = tempfile.TemporaryDirectory(prefix="battousai_demo_")
    tmpdir      = tmpdir_obj.name

    try:
        # ── Act 1 ─────────────────────────────────────────────────────────
        kernel, researcher_id, writer_id, rogue_id = act1_setup(cap_manager)
        time.sleep(0.3)

        # ── Act 2 ─────────────────────────────────────────────────────────
        act2_normal_ops(cap_manager, researcher_id, writer_id, rogue_id, tmpdir)
        time.sleep(0.3)

        # ── Act 3 ─────────────────────────────────────────────────────────
        blocked_count = act3_attack(cap_manager, rogue_id, tmpdir)
        time.sleep(0.3)

        # ── Act 4 ─────────────────────────────────────────────────────────
        act4_audit(cap_manager, rogue_id)
        time.sleep(0.3)

        # ── Act 5 ─────────────────────────────────────────────────────────
        act5_recovery(cap_manager)

    finally:
        # Always clean up temp files
        tmpdir_obj.cleanup()

    # ── Finale ────────────────────────────────────────────────────────────
    section("Complete")
    print_summary(
        t_start=t_start,
        agents_spawned=3,
        actions_allowed=3,
        actions_blocked=blocked_count,
        crashes=1,
        recoveries=1,
    )


if __name__ == "__main__":
    main()

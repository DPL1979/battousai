"""
main.py — Battousai Demo Entry Point
=================================
Boots the Autonomous Intelligence Operating System and runs a multi-agent
demonstration scenario:

    Scenario: "Research and Summarise Quantum Computing"
    ────────────────────────────────────────────────────
    1. A CoordinatorAgent receives the top-level research task
    2. The Coordinator decomposes it into two parallel subtasks and
       spawns two WorkerAgents to handle them
    3. Worker-1 uses the `web_search` tool to gather fundamentals,
       milestones, and challenges
    4. Worker-2 uses the `web_search` tool to gather applications
       and future outlook
    5. Both workers send their results back to the Coordinator
    6. The Coordinator synthesises the results and writes a summary
       to /shared/results/summary.txt in the virtual filesystem
    7. A MonitorAgent watches system health throughout and publishes
       periodic metrics snapshots
    8. At the end, a full system report is printed

Usage:
    python -m battousai.main                  # default 50 ticks
    python -m battousai.main --ticks 80       # custom tick count
    python -m battousai.main --debug          # verbose debug output

Exit codes:
    0  — completed successfully
    1  — kernel panic or uncaught exception
"""

from __future__ import annotations

import argparse
import sys
import time

from battousai.agent import CoordinatorAgent, MonitorAgent, WorkerAgent
from battousai.ipc import MessageType
from battousai.kernel import Kernel, KernelPanic
from battousai.logger import LogLevel


BANNER = r"""
 ██████╗  █████╗ ████████╗████████╗ ██████╗ ██╗   ██╗███████╗ █████╗ ██╗
 ██╔══██╗██╔══██╗╚══██╔══╝╚══██╔══╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██║
 ██████╔╝███████║   ██║      ██║   ██║   ██║██║   ██║███████╗███████║██║
 ██╔══██╗██╔══██║   ██║      ██║   ██║   ██║██║   ██║╚════██║██╔══██║██║
 ██████╔╝██║  ██║   ██║      ██║   ╚██████╔╝╚██████╔╝███████║██║  ██║██║
 ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝
 Autonomous Intelligence Operating System  v0.2.0
 An OS designed exclusively for AI agents.
 No humans. No GUI. No terminal. Agents are first-class citizens.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Battousai — Autonomous Intelligence Operating System demo"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=50,
        help="Number of simulation ticks to run (default: 50)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG-level log output",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        default=False,
        help="Suppress the ASCII art banner",
    )
    return parser.parse_args()


def run_demo(ticks: int = 50, debug: bool = False, show_banner: bool = True) -> int:
    """
    Execute the Battousai demonstration scenario.

    Returns 0 on success, 1 on failure.
    """
    if show_banner:
        print(BANNER)

    wall_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Create and boot the kernel
    # -------------------------------------------------------------------------
    kernel = Kernel(max_ticks=ticks, debug=debug)

    try:
        kernel.boot()
    except KernelPanic as exc:
        print(f"\n[KERNEL PANIC] {exc}", file=sys.stderr)
        return 1

    # -------------------------------------------------------------------------
    # 2. Spawn the initial agent set
    #    MonitorAgent first (highest priority in its band) so it can observe
    #    the full lifecycle from tick 1.
    # -------------------------------------------------------------------------

    kernel.logger.system("main", "Spawning initial agents...")

    # MonitorAgent: low priority (7), observes passively
    monitor_id = kernel.spawn_agent(
        MonitorAgent,
        name="SysMonitor",
        priority=7,
    )

    # CoordinatorAgent: moderate-high priority (2)
    coordinator_id = kernel.spawn_agent(
        CoordinatorAgent,
        name="Coordinator",
        priority=2,
    )

    # WorkerAgents will be spawned by the Coordinator on demand.
    # We pre-register tool access for all current and future agents.
    for tool_name in ["web_search", "calculator", "code_executor", "file_reader", "file_writer"]:
        kernel.tools.get_spec(tool_name).allowed_agents = set()  # Open access

    # -------------------------------------------------------------------------
    # 3. Seed the Coordinator with the initial task via the kernel's IPC layer
    # -------------------------------------------------------------------------
    kernel.logger.system("main", "Delivering initial task to Coordinator...")

    seed_msg = kernel.ipc.create_message(
        sender_id="kernel",
        recipient_id=coordinator_id,
        message_type=MessageType.TASK,
        payload={"task": "Research and summarize quantum computing"},
        timestamp=0,
    )
    kernel.logger.info("main", f"Task message {seed_msg.message_id} delivered to {coordinator_id}")

    # -------------------------------------------------------------------------
    # 4. Run the event loop
    # -------------------------------------------------------------------------
    kernel.logger.system("main", f"Starting event loop ({ticks} ticks)...")

    try:
        kernel.run()
    except KeyboardInterrupt:
        kernel.logger.warn("main", "Interrupted by KeyboardInterrupt")
    except Exception as exc:
        kernel.logger.error("main", f"Event loop crashed: {exc}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1

    wall_elapsed = time.time() - wall_start

    # -------------------------------------------------------------------------
    # 5. Retrieve and display the MonitorAgent's report
    # -------------------------------------------------------------------------
    # The monitor agent may have been GC'd at end of run, get its ref
    monitor_agent = kernel._agents.get(monitor_id)
    if monitor_agent and isinstance(monitor_agent, MonitorAgent):
        print("\n" + monitor_agent.get_report())

    # -------------------------------------------------------------------------
    # 6. Print the full system report
    # -------------------------------------------------------------------------
    report = kernel.system_report()
    print(report)
    print(f"\n  Wall-clock time: {wall_elapsed:.2f}s for {ticks} ticks")
    print(f"  Tick rate: {ticks / wall_elapsed:.0f} ticks/sec\n")

    return 0


def main() -> None:
    args = parse_args()
    exit_code = run_demo(
        ticks=args.ticks,
        debug=args.debug,
        show_banner=not args.no_banner,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

"""
quickstart.py — Battousai Hello World
======================================
Minimal example that shows how to:

  1. Boot a Battousai kernel
  2. Spawn an LLMAgent backed by MockLLMProvider
  3. Run for 5 ticks
  4. Print the system report

Run:  python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Ensure workspace root is on the path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battousai.kernel import Kernel
from battousai.llm import LLMAgent, MockLLMProvider, LLMRouter
from battousai.logger import LogLevel

# ---------------------------------------------------------------------------
# 1. Boot the kernel
# ---------------------------------------------------------------------------
print("\n[quickstart] Booting Battousai kernel…")
kernel = Kernel(max_ticks=5, debug=False)
# Keep console output minimal — show only INFO+ from the kernel itself
kernel.logger.min_level = LogLevel.WARN
kernel.logger.console_output = False
kernel.boot()
print("[quickstart] Kernel online ✓")

# ---------------------------------------------------------------------------
# 2. Register a MockLLMProvider and build an LLMRouter
# ---------------------------------------------------------------------------
router = LLMRouter()
router.register_provider("mock", MockLLMProvider(model_name="mock-gpt-1"))
router.set_default("mock")

print("[quickstart] LLMRouter configured with MockLLMProvider ✓")

# ---------------------------------------------------------------------------
# 3. Spawn an LLMAgent
# ---------------------------------------------------------------------------
agent_id = kernel.spawn_agent(
    LLMAgent,
    name="HelloAgent",
    priority=4,
    provider_name="mock",
    system_prompt="You are a concise assistant inside the Battousai agent OS.",
    llm_router=router,
)
print(f"[quickstart] LLMAgent spawned → {agent_id!r} ✓")

# ---------------------------------------------------------------------------
# 4. Run for 5 ticks
# ---------------------------------------------------------------------------
print("[quickstart] Running 5 ticks…")
kernel.run(ticks=5)
print("[quickstart] Ticks complete ✓")

# ---------------------------------------------------------------------------
# 5. Print the system report
# ---------------------------------------------------------------------------
report = kernel.system_report()
print(report)

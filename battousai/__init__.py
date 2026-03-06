"""
Battousai - Autonomous Intelligence Operating System
================================================
An operating system designed exclusively for AI agents.
No human users. No GUI. No terminal. Agents are first-class citizens.

Version: 0.6.0

New in v0.6.0:
    - OS-level process sandbox (sandbox.py): seccomp-BPF, Linux namespaces,
      resource limits, environment sanitization; SandboxedProcess, SandboxProfile,
      predefined profiles (MINIMAL/STANDARD/NETWORK/PRIVILEGED), capability_to_profile

New in v0.5.0:
    - Human-in-the-Loop Approval Workflow (approval.py): risk tiers, approval
      gate, middleware integration with the kernel's syscall path

New in v0.4.0:
    - MCP Server adapter (mcp_server.py): expose Battousai tools as an MCP server
    - MCP Client adapter (mcp_client.py): connect to external MCP servers

New in v0.3.0:
    - Real LLM providers (providers.py): OpenAI, Anthropic, Ollama — zero deps
    - Sandboxed filesystem (real_fs.py): real OS I/O with per-agent jails
    - SQLite persistence (persistence.py): agent state, memory, audit logs
    - Process isolation (isolation.py): subprocess-based agent sandboxing

New in v0.2.0:
    - LLM integration (llm.py): LLMProvider, LLMAgent, ContextWindow
    - Supervision trees (supervisor.py): Erlang/OTP-style fault tolerance
    - Extended tools (tools_extended.py): 9 new tools including vector_store,
      python_repl, task_queue, and data_pipeline
    - Capability-based security (capabilities.py): least-privilege token system
    - Typed memory schemas (schemas.py): runtime-validated memory contracts
    - Network stack (network.py): gossip, service discovery, agent migration
    - Multi-kernel federation (federation.py): Raft consensus, load balancing
    - Self-modification & evolution (evolution.py): code sandbox, genetic pool
    - Hardware abstraction layer (hal.py): simulated GPIO, sensors, cameras
    - Formal verification & contracts (contracts.py): Design-by-Contract runtime
"""

__version__ = "0.3.0"
__author__ = "Battousai Project"
__description__ = "Autonomous Intelligence Operating System — an OS built for AI agents"

# ---------------------------------------------------------------------------
# Core runtime
# ---------------------------------------------------------------------------
from battousai.kernel import Kernel
from battousai.logger import Logger, LogLevel

# ---------------------------------------------------------------------------
# Agent base classes
# ---------------------------------------------------------------------------
from battousai.agent import Agent, CoordinatorAgent, WorkerAgent, MonitorAgent

# ---------------------------------------------------------------------------
# LLM integration (v0.2.0)
# ---------------------------------------------------------------------------
from battousai.llm import LLMAgent, LLMRouter, MockLLMProvider, ContextWindow

# ---------------------------------------------------------------------------
# Supervision trees (v0.2.0)
# ---------------------------------------------------------------------------
from battousai.supervisor import (
    SupervisorAgent,
    ChildSpec,
    RestartStrategy,
    RestartType,
)

# ---------------------------------------------------------------------------
# Security layer (v0.2.0)
# ---------------------------------------------------------------------------
from battousai.capabilities import CapabilityManager, CapabilityType, CapabilityViolation
from battousai.contracts import ContractMonitor, Contract, SafetyEnvelope

# ---------------------------------------------------------------------------
# Real I/O layer (v0.3.0)
# ---------------------------------------------------------------------------
from battousai.providers import (
    RealOpenAIProvider,
    RealAnthropicProvider,
    OllamaProvider,
    HTTPProvider,
    LLMProviderError,
    AuthenticationError,
    RateLimitError,
)
from battousai.real_fs import SandboxedFilesystem
from battousai.persistence import PersistenceLayer
from battousai.isolation import IsolatedAgentProcess, ProcessPool, SandboxConfig

# ---------------------------------------------------------------------------
# MCP adapters (v0.4.0)
# ---------------------------------------------------------------------------
from battousai.mcp_server import MCPServer, MCPServerConfig, MCPProtocolError
from battousai.mcp_client import MCPClient, MCPClientConfig, MCPConnectionError

# ---------------------------------------------------------------------------
# Human-in-the-Loop Approval Workflow (v0.5.0)
# ---------------------------------------------------------------------------
from battousai.approval import (
    ApprovalGate,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
    ApprovalMiddleware,
    ApprovalHandler,
    AutoApproveHandler,
    CallbackApprovalHandler,
    CLIApprovalHandler,
    RiskTier,
    DEFAULT_RISK_MAP,
)

# ---------------------------------------------------------------------------
# Memory Integrity (v0.5.0)
# ---------------------------------------------------------------------------
from battousai.integrity import (
    HashChain,
    SecureMemoryStore,
    ToolRegistryVerifier,
    IntegrityAuditor,
    HashChainEntry,
    IntegrityReport,
    MemoryEntry as SecureMemoryEntry,
    RegistrySignature,
    AuditResult,
    IntegrityError,
    IntegrityViolation,
    EntryNotFoundError,
    EntryExpiredError,
)

# ---------------------------------------------------------------------------
# OS-Level Sandbox (v0.6.0)
# ---------------------------------------------------------------------------
from battousai.sandbox import (
    SandboxedProcess,
    SandboxProfile,
    PROFILE_MINIMAL,
    PROFILE_STANDARD,
    PROFILE_NETWORK,
    PROFILE_PRIVILEGED,
    EnvironmentSanitizer,
    SeccompFilter,
    NamespaceIsolation,
    ResourceLimiter,
    EnforcementReport,
    SandboxResult,
    capability_to_profile,
)

__all__ = [
    # Core
    "Kernel",
    "Logger",
    "LogLevel",
    # Agent base classes
    "Agent",
    "CoordinatorAgent",
    "WorkerAgent",
    "MonitorAgent",
    # LLM integration
    "LLMAgent",
    "LLMRouter",
    "MockLLMProvider",
    "ContextWindow",
    # Supervision
    "SupervisorAgent",
    "ChildSpec",
    "RestartStrategy",
    "RestartType",
    # Security
    "CapabilityManager",
    "CapabilityType",
    "CapabilityViolation",
    "ContractMonitor",
    "Contract",
    "SafetyEnvelope",
    # Real I/O (v0.3.0)
    "RealOpenAIProvider",
    "RealAnthropicProvider",
    "OllamaProvider",
    "HTTPProvider",
    "LLMProviderError",
    "AuthenticationError",
    "RateLimitError",
    "SandboxedFilesystem",
    "PersistenceLayer",
    "IsolatedAgentProcess",
    "ProcessPool",
    "SandboxConfig",
    # MCP adapters (v0.4.0)
    "MCPServer",
    "MCPServerConfig",
    "MCPProtocolError",
    "MCPClient",
    "MCPClientConfig",
    "MCPConnectionError",
    # Approval workflow (v0.5.0)
    "ApprovalGate",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalResult",
    "ApprovalMiddleware",
    "ApprovalHandler",
    "AutoApproveHandler",
    "CallbackApprovalHandler",
    "CLIApprovalHandler",
    "RiskTier",
    "DEFAULT_RISK_MAP",
    # Memory Integrity (v0.5.0)
    "HashChain",
    "SecureMemoryStore",
    "ToolRegistryVerifier",
    "IntegrityAuditor",
    "HashChainEntry",
    "IntegrityReport",
    "SecureMemoryEntry",
    "RegistrySignature",
    "AuditResult",
    "IntegrityError",
    "IntegrityViolation",
    "EntryNotFoundError",
    "EntryExpiredError",
    # OS-Level Sandbox (v0.6.0)
    "SandboxedProcess",
    "SandboxProfile",
    "PROFILE_MINIMAL",
    "PROFILE_STANDARD",
    "PROFILE_NETWORK",
    "PROFILE_PRIVILEGED",
    "EnvironmentSanitizer",
    "SeccompFilter",
    "NamespaceIsolation",
    "ResourceLimiter",
    "EnforcementReport",
    "SandboxResult",
    "capability_to_profile",
]


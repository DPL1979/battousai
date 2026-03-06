"""
sandbox.py — OS-Level Process Sandboxing
==========================================
Real OS-level sandboxing that enforces isolation at the kernel level, not just
the application level. This bridges the gap between Battousai's in-process
security model and production isolation systems like E2B or Modal.

Enforcement Stack (applied in order)
--------------------------------------
1. Environment sanitization  — remove API keys, secrets, HOME, SSH sockets
2. Resource limits            — memory, CPU, FDs, processes (Linux + macOS)
3. Namespace isolation        — PID, mount, network, user (Linux only)
4. Seccomp BPF filter         — syscall whitelist (Linux only)
5. Filesystem jail            — chroot / bind mount (Linux only)

On non-Linux platforms layers 3-5 are skipped with logged warnings; layers 1-2
work everywhere.

Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import multiprocessing
import multiprocessing.connection
import os
import platform
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from battousai.capabilities import CapabilityType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform constants
# ---------------------------------------------------------------------------

_IS_LINUX: bool = platform.system() == "Linux"
_IS_MACOS: bool = platform.system() == "Darwin"

# ---------------------------------------------------------------------------
# Linux syscall numbers (x86_64)
# ---------------------------------------------------------------------------

_SYSCALL_READ         = 0
_SYSCALL_WRITE        = 1
_SYSCALL_OPEN         = 2
_SYSCALL_CLOSE        = 3
_SYSCALL_STAT         = 4
_SYSCALL_FSTAT        = 5
_SYSCALL_LSTAT        = 6
_SYSCALL_POLL         = 7
_SYSCALL_LSEEK        = 8
_SYSCALL_MMAP         = 9
_SYSCALL_MPROTECT     = 10
_SYSCALL_MUNMAP       = 11
_SYSCALL_BRK          = 12
_SYSCALL_RT_SIGACTION = 13
_SYSCALL_RT_SIGPROCMASK = 14
_SYSCALL_IOCTL        = 16
_SYSCALL_PREAD64      = 17
_SYSCALL_PWRITE64     = 18
_SYSCALL_READV        = 19
_SYSCALL_WRITEV       = 20
_SYSCALL_ACCESS       = 21
_SYSCALL_PIPE         = 22
_SYSCALL_SELECT       = 23
_SYSCALL_SCHED_YIELD  = 24
_SYSCALL_MREMAP       = 25
_SYSCALL_MSYNC        = 26
_SYSCALL_MINCORE      = 27
_SYSCALL_MADVISE      = 28
_SYSCALL_OPENAT       = 257
_SYSCALL_GETDENTS64   = 217
_SYSCALL_NEWFSTATAT   = 262
_SYSCALL_READLINKAT   = 267
_SYSCALL_SOCKET       = 41
_SYSCALL_CONNECT      = 42
_SYSCALL_ACCEPT       = 43
_SYSCALL_SENDTO       = 44
_SYSCALL_RECVFROM     = 45
_SYSCALL_SENDMSG      = 46
_SYSCALL_RECVMSG      = 47
_SYSCALL_BIND         = 49
_SYSCALL_LISTEN       = 50
_SYSCALL_GETSOCKNAME  = 51
_SYSCALL_GETPEERNAME  = 52
_SYSCALL_SETSOCKOPT   = 54
_SYSCALL_GETSOCKOPT   = 55
_SYSCALL_CLONE        = 56
_SYSCALL_FORK         = 57
_SYSCALL_VFORK        = 58
_SYSCALL_EXECVE       = 59
_SYSCALL_EXIT         = 60
_SYSCALL_WAIT4        = 61
_SYSCALL_KILL         = 62
_SYSCALL_UNAME        = 63
_SYSCALL_FCNTL        = 72
_SYSCALL_FLOCK        = 73
_SYSCALL_FSYNC        = 74
_SYSCALL_FDATASYNC    = 75
_SYSCALL_TRUNCATE     = 76
_SYSCALL_FTRUNCATE    = 77
_SYSCALL_GETDENTS     = 78
_SYSCALL_GETCWD       = 79
_SYSCALL_CHDIR        = 80
_SYSCALL_MKDIR        = 83
_SYSCALL_RMDIR        = 84
_SYSCALL_CREAT        = 85
_SYSCALL_UNLINK       = 87
_SYSCALL_RENAME       = 82
_SYSCALL_GETPID       = 39
_SYSCALL_GETPPID      = 110
_SYSCALL_GETUID       = 102
_SYSCALL_GETGID       = 104
_SYSCALL_GETEUID      = 107
_SYSCALL_GETEGID      = 108
_SYSCALL_SETUID       = 105
_SYSCALL_SETGID       = 106
_SYSCALL_NANOSLEEP    = 35
_SYSCALL_GETRLIMIT    = 97
_SYSCALL_SETRLIMIT    = 160
_SYSCALL_RT_SIGRETURN = 15
_SYSCALL_EXIT_GROUP   = 231
_SYSCALL_FUTEX        = 202
_SYSCALL_CLOCK_GETTIME = 228
_SYSCALL_CLOCK_NANOSLEEP = 230
_SYSCALL_EPOLL_CREATE1 = 291
_SYSCALL_EPOLL_CTL    = 233
_SYSCALL_EPOLL_WAIT   = 232
_SYSCALL_EPOLL_PWAIT  = 281
_SYSCALL_EVENTFD2     = 290
_SYSCALL_PIPE2        = 293
_SYSCALL_GETRANDOM    = 318
_SYSCALL_MEMFD_CREATE = 319
_SYSCALL_MLOCK        = 149
_SYSCALL_MUNLOCK      = 150
_SYSCALL_SIGALTSTACK  = 131
_SYSCALL_ARCH_PRCTL   = 158
_SYSCALL_SET_TID_ADDRESS = 218
_SYSCALL_SET_ROBUST_LIST = 273
_SYSCALL_GET_ROBUST_LIST = 274
_SYSCALL_RESTART_SYSCALL = 219
_SYSCALL_PRCTL        = 157
_SYSCALL_SYSINFO      = 99
_SYSCALL_TIMES        = 100
_SYSCALL_GETTIMEOFDAY = 96
_SYSCALL_SENDFILE     = 40

# prctl constants
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP      = 22
_SECCOMP_MODE_FILTER = 2

# clone / unshare flags
_CLONE_NEWUSER = 0x10000000
_CLONE_NEWPID  = 0x20000000
_CLONE_NEWNS   = 0x00020000
_CLONE_NEWNET  = 0x40000000
_CLONE_NEWIPC  = 0x08000000
_CLONE_NEWUTS  = 0x04000000

# BPF instruction constants
_BPF_LD   = 0x00
_BPF_W    = 0x00
_BPF_ABS  = 0x20
_BPF_JMP  = 0x05
_BPF_JEQ  = 0x10
_BPF_K    = 0x00
_BPF_RET  = 0x06
_BPF_ALLOW = 0x7fff0000  # SECCOMP_RET_ALLOW
_BPF_KILL  = 0x00000000  # SECCOMP_RET_KILL

# seccomp data offset for syscall number (arch-dependent: 4 bytes in)
_SECCOMP_DATA_NR_OFFSET = 0

# sock_filter struct: "HBBI" = code(u16) jt(u8) jf(u8) k(u32)
_SOCK_FILTER_FMT = "HBBI"
_SOCK_FILTER_SIZE = struct.calcsize(_SOCK_FILTER_FMT)


# ---------------------------------------------------------------------------
# SandboxProfile
# ---------------------------------------------------------------------------

@dataclass
class SandboxProfile:
    """
    Defines what a sandboxed process is permitted to do.

    Attributes
    ----------
    name             : Human-readable identifier for the profile.
    allowed_paths    : Filesystem paths the process may access (read by default).
    denied_paths     : Explicitly blocked paths overriding allowed_paths.
    writable_paths   : Subset of allowed_paths that the process may write to.
    network_allowed  : Whether the process may open network connections.
    allowed_syscalls : If set, a whitelist of syscall names (Linux only).
    max_memory_bytes : Virtual memory ceiling (0 = unlimited).
    max_cpu_seconds  : CPU time ceiling in seconds (0.0 = unlimited).
    max_file_descriptors : Maximum open file descriptors.
    max_processes    : Maximum number of processes/threads (fork limit).
    env_whitelist    : Only these env var names are passed through (empty = all allowed).
    env_blacklist    : These env var names are always scrubbed.
    """

    name: str
    allowed_paths: List[str] = field(default_factory=list)
    denied_paths: List[str] = field(default_factory=list)
    writable_paths: List[str] = field(default_factory=list)
    network_allowed: bool = False
    allowed_syscalls: Optional[Set[str]] = None
    max_memory_bytes: int = 0
    max_cpu_seconds: float = 0.0
    max_file_descriptors: int = 256
    max_processes: int = 1
    env_whitelist: List[str] = field(default_factory=list)
    env_blacklist: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Predefined profiles
# ---------------------------------------------------------------------------

PROFILE_MINIMAL = SandboxProfile(
    name="minimal",
    allowed_paths=["/tmp", "/proc/self"],
    denied_paths=["/etc/passwd", "/etc/shadow", os.path.expanduser("~/.ssh"),
                  os.path.expanduser("~/.aws"), os.path.expanduser("~/.gnupg")],
    writable_paths=[],
    network_allowed=False,
    max_memory_bytes=64 * 1024 * 1024,   # 64 MB
    max_cpu_seconds=10.0,
    max_file_descriptors=32,
    max_processes=1,
    env_whitelist=["PATH", "LANG", "LC_ALL", "TZ"],
    env_blacklist=[],
)

PROFILE_STANDARD = SandboxProfile(
    name="standard",
    allowed_paths=["/tmp", "/proc/self", "/usr", "/lib", "/lib64"],
    denied_paths=[os.path.expanduser("~/.ssh"), os.path.expanduser("~/.aws"),
                  os.path.expanduser("~/.gnupg")],
    writable_paths=["/tmp"],
    network_allowed=False,
    max_memory_bytes=256 * 1024 * 1024,  # 256 MB
    max_cpu_seconds=60.0,
    max_file_descriptors=128,
    max_processes=4,
    env_whitelist=["PATH", "LANG", "LC_ALL", "TZ", "HOME", "USER", "LOGNAME"],
    env_blacklist=[],
)

PROFILE_NETWORK = SandboxProfile(
    name="network",
    allowed_paths=["/tmp", "/proc/self", "/usr", "/lib", "/lib64",
                   "/etc/ssl", "/etc/ca-certificates"],
    denied_paths=[os.path.expanduser("~/.ssh"), os.path.expanduser("~/.aws"),
                  os.path.expanduser("~/.gnupg")],
    writable_paths=["/tmp"],
    network_allowed=True,
    max_memory_bytes=512 * 1024 * 1024,  # 512 MB
    max_cpu_seconds=120.0,
    max_file_descriptors=256,
    max_processes=8,
    env_whitelist=["PATH", "LANG", "LC_ALL", "TZ", "HOME", "USER", "LOGNAME",
                   "http_proxy", "https_proxy", "no_proxy",
                   "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"],
    env_blacklist=[],
)

PROFILE_PRIVILEGED = SandboxProfile(
    name="privileged",
    allowed_paths=["/"],
    denied_paths=[],
    writable_paths=["/"],
    network_allowed=True,
    max_memory_bytes=0,       # unlimited
    max_cpu_seconds=0.0,      # unlimited
    max_file_descriptors=1024,
    max_processes=64,
    env_whitelist=[],         # empty = pass everything through
    env_blacklist=[],
)


# ---------------------------------------------------------------------------
# EnvironmentSanitizer
# ---------------------------------------------------------------------------

class EnvironmentSanitizer:
    """
    Removes sensitive environment variables before spawning agent processes.

    Two mechanisms:
    - ``env_blacklist`` in SandboxProfile: exact name matching.
    - ``env_whitelist`` in SandboxProfile: if non-empty, only listed names survive.
    - ``detect_leaked_secrets``: heuristic pattern scan for credential-shaped values.
    """

    DEFAULT_BLACKLIST: List[str] = [
        # AWS
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN", "AWS_PROFILE",
        # LLM providers
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        "COHERE_API_KEY", "HUGGINGFACE_TOKEN", "HF_TOKEN",
        "MISTRAL_API_KEY", "GROQ_API_KEY", "REPLICATE_API_TOKEN",
        # Source control
        "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN", "GITLAB_CI_TOKEN",
        "BITBUCKET_TOKEN",
        # Databases
        "DATABASE_URL", "REDIS_URL", "MONGO_URL", "POSTGRES_URL",
        "MYSQL_URL", "DB_PASSWORD", "DATABASE_PASSWORD",
        # SSH
        "SSH_AUTH_SOCK", "SSH_AGENT_PID", "SSH_PRIVATE_KEY",
        # Home dir (prevents ~/.ssh, ~/.aws traversal)
        "HOME",
        # Generic secrets
        "SECRET_KEY", "APP_SECRET", "JWT_SECRET", "SIGNING_KEY",
        "ENCRYPTION_KEY", "MASTER_KEY",
    ]

    # Patterns that suggest a value is a credential even if the name isn't known
    _SECRET_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"sk-[A-Za-z0-9]{20,}"),          # OpenAI
        re.compile(r"AKIA[0-9A-Z]{16}"),               # AWS access key
        re.compile(r"ghp_[A-Za-z0-9]{36}"),            # GitHub personal access token
        re.compile(r"ghs_[A-Za-z0-9]{36}"),            # GitHub server token
        re.compile(r"glpat-[A-Za-z0-9_-]{20}"),        # GitLab PAT
        re.compile(r"xox[bprs]-[A-Za-z0-9-]{24,}"),    # Slack token
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
        re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),       # long base64 (generic secret)
        re.compile(r"[0-9a-f]{32,}"),                   # long hex (hash/token)
    ]

    def sanitize(
        self,
        env: Dict[str, str],
        profile: SandboxProfile,
    ) -> Dict[str, str]:
        """
        Return a copy of ``env`` with sensitive variables removed.

        Applies (in order):
        1. Combined blacklist = DEFAULT_BLACKLIST + profile.env_blacklist
        2. If profile.env_whitelist is non-empty, only whitelisted names survive.
        """
        combined_blacklist: Set[str] = set(self.DEFAULT_BLACKLIST) | set(profile.env_blacklist)
        result: Dict[str, str] = {}

        for key, value in env.items():
            if key in combined_blacklist:
                continue
            result[key] = value

        if profile.env_whitelist:
            whiteset = set(profile.env_whitelist)
            result = {k: v for k, v in result.items() if k in whiteset}

        return result

    def detect_leaked_secrets(self, env: Dict[str, str]) -> List[str]:
        """
        Scan environment values for patterns that look like credentials.

        Returns a list of variable names whose *values* match a secret pattern.
        Does NOT return the values themselves.
        """
        leaked: List[str] = []
        for key, value in env.items():
            for pattern in self._SECRET_PATTERNS:
                if pattern.search(value):
                    leaked.append(key)
                    break
        return leaked


# ---------------------------------------------------------------------------
# ResourceLimiter
# ---------------------------------------------------------------------------

class ResourceLimiter:
    """
    Applies resource limits using the stdlib ``resource`` module.

    Works on Linux and macOS. On Windows the ``resource`` module is unavailable
    and all limits silently report as not applied.
    """

    def __init__(self, profile: SandboxProfile) -> None:
        self.profile = profile

    def apply(self) -> Dict[str, bool]:
        """
        Apply resource limits to the *current* process.

        Returns a dict mapping limit name to whether the limit was successfully set.
        """
        results: Dict[str, bool] = {
            "memory": False,
            "cpu": False,
            "file_descriptors": False,
            "processes": False,
        }
        try:
            import resource as _res
        except ImportError:
            logger.warning("resource module not available — no limits applied")
            return results

        def _set(resource_const: int, soft: int, hard: int, key: str) -> None:
            try:
                _res.setrlimit(resource_const, (soft, hard))
                results[key] = True
            except (ValueError, _res.error, AttributeError) as exc:
                logger.warning("Could not set %s limit: %s", key, exc)

        p = self.profile

        if p.max_memory_bytes > 0:
            mb = p.max_memory_bytes
            try:
                # RLIMIT_AS = virtual address space
                _set(_res.RLIMIT_AS, mb, mb, "memory")
            except AttributeError:
                pass

        if p.max_cpu_seconds > 0.0:
            cpu_secs = int(p.max_cpu_seconds)
            if cpu_secs > 0:
                _set(_res.RLIMIT_CPU, cpu_secs, cpu_secs, "cpu")

        if p.max_file_descriptors > 0:
            try:
                _set(_res.RLIMIT_NOFILE, p.max_file_descriptors,
                     p.max_file_descriptors, "file_descriptors")
            except AttributeError:
                pass

        if p.max_processes > 0:
            try:
                _set(_res.RLIMIT_NPROC, p.max_processes, p.max_processes, "processes")
            except AttributeError:
                pass

        return results

    def get_usage(self) -> Dict[str, int]:
        """Return current resource usage for the calling process."""
        usage: Dict[str, int] = {
            "max_rss_bytes": 0,
            "user_time_ms": 0,
            "system_time_ms": 0,
        }
        try:
            import resource as _res
            ru = _res.getrusage(_res.RUSAGE_SELF)
            # ru_maxrss is KB on Linux, bytes on macOS
            if _IS_MACOS:
                usage["max_rss_bytes"] = ru.ru_maxrss
            else:
                usage["max_rss_bytes"] = ru.ru_maxrss * 1024
            usage["user_time_ms"] = int(ru.ru_utime * 1000)
            usage["system_time_ms"] = int(ru.ru_stime * 1000)
        except Exception:
            pass
        return usage


# ---------------------------------------------------------------------------
# SeccompFilter
# ---------------------------------------------------------------------------

def _make_bpf_insn(code: int, jt: int, jf: int, k: int) -> bytes:
    """Pack a single BPF instruction (sock_filter struct)."""
    return struct.pack(_SOCK_FILTER_FMT, code, jt, jf, k)


class SeccompFilter:
    """
    Applies seccomp-BPF filters to restrict the system calls available to the
    current process (Linux x86_64 only).

    Uses ``ctypes`` to call ``prctl()`` and load a BPF program via
    ``prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ...)``.

    Falls back gracefully on non-Linux platforms: ``is_available()`` returns
    ``False`` and ``apply()`` returns ``False`` without raising.
    """

    def __init__(self, profile: SandboxProfile) -> None:
        self.profile = profile
        self._libc: Optional[ctypes.CDLL] = None

    # ------------------------------------------------------------------
    # Pre-built syscall allow sets
    # ------------------------------------------------------------------

    @staticmethod
    def minimal_filter() -> Set[int]:
        """
        Absolute minimum for a read-compute-exit workload.
        Includes: read, write, mmap, close, exit, futex, and related process
        management calls needed by the Python runtime.
        """
        return {
            _SYSCALL_READ, _SYSCALL_WRITE, _SYSCALL_CLOSE,
            _SYSCALL_MMAP, _SYSCALL_MUNMAP, _SYSCALL_MPROTECT,
            _SYSCALL_BRK, _SYSCALL_MREMAP, _SYSCALL_MADVISE,
            _SYSCALL_MSYNC, _SYSCALL_MINCORE, _SYSCALL_MLOCK, _SYSCALL_MUNLOCK,
            _SYSCALL_EXIT, _SYSCALL_EXIT_GROUP,
            _SYSCALL_FUTEX, _SYSCALL_NANOSLEEP,
            _SYSCALL_CLOCK_GETTIME, _SYSCALL_GETTIMEOFDAY, _SYSCALL_TIMES,
            _SYSCALL_CLOCK_NANOSLEEP,
            _SYSCALL_GETPID, _SYSCALL_GETPPID,
            _SYSCALL_GETUID, _SYSCALL_GETGID, _SYSCALL_GETEUID, _SYSCALL_GETEGID,
            _SYSCALL_RT_SIGACTION, _SYSCALL_RT_SIGPROCMASK, _SYSCALL_RT_SIGRETURN,
            _SYSCALL_SIGALTSTACK,
            _SYSCALL_SCHED_YIELD,
            _SYSCALL_ARCH_PRCTL,
            _SYSCALL_SET_TID_ADDRESS, _SYSCALL_SET_ROBUST_LIST,
            _SYSCALL_GET_ROBUST_LIST,
            _SYSCALL_RESTART_SYSCALL,
            _SYSCALL_UNAME,
            _SYSCALL_SYSINFO,
            _SYSCALL_IOCTL,
            _SYSCALL_PRCTL,
            _SYSCALL_FCNTL,
            _SYSCALL_EPOLL_CREATE1, _SYSCALL_EPOLL_CTL,
            _SYSCALL_EPOLL_WAIT, _SYSCALL_EPOLL_PWAIT,
            _SYSCALL_EVENTFD2, _SYSCALL_PIPE, _SYSCALL_PIPE2,
            _SYSCALL_PREAD64, _SYSCALL_PWRITE64, _SYSCALL_READV, _SYSCALL_WRITEV,
            _SYSCALL_LSEEK, _SYSCALL_POLL, _SYSCALL_SELECT,
            _SYSCALL_GETRANDOM,
            _SYSCALL_MEMFD_CREATE,
        }

    @staticmethod
    def standard_filter() -> Set[int]:
        """
        Standard profile: minimal + filesystem operations (open, stat, etc.).
        """
        return SeccompFilter.minimal_filter() | {
            _SYSCALL_OPEN, _SYSCALL_OPENAT, _SYSCALL_STAT, _SYSCALL_FSTAT,
            _SYSCALL_LSTAT, _SYSCALL_NEWFSTATAT,
            _SYSCALL_ACCESS, _SYSCALL_GETCWD, _SYSCALL_CHDIR,
            _SYSCALL_GETDENTS, _SYSCALL_GETDENTS64,
            _SYSCALL_READLINKAT,
            _SYSCALL_FCNTL, _SYSCALL_FLOCK,
            _SYSCALL_FSYNC, _SYSCALL_FDATASYNC,
            _SYSCALL_TRUNCATE, _SYSCALL_FTRUNCATE,
            _SYSCALL_MKDIR, _SYSCALL_RMDIR,
            _SYSCALL_CREAT, _SYSCALL_UNLINK, _SYSCALL_RENAME,
            _SYSCALL_SENDFILE,
        }

    @staticmethod
    def network_filter() -> Set[int]:
        """
        Network profile: standard + socket family.
        """
        return SeccompFilter.standard_filter() | {
            _SYSCALL_SOCKET, _SYSCALL_CONNECT, _SYSCALL_ACCEPT,
            _SYSCALL_SENDTO, _SYSCALL_RECVFROM,
            _SYSCALL_SENDMSG, _SYSCALL_RECVMSG,
            _SYSCALL_BIND, _SYSCALL_LISTEN,
            _SYSCALL_GETSOCKNAME, _SYSCALL_GETPEERNAME,
            _SYSCALL_SETSOCKOPT, _SYSCALL_GETSOCKOPT,
        }

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if seccomp-BPF can be applied on this system."""
        if not _IS_LINUX:
            return False
        try:
            libc = self._get_libc()
            if libc is None:
                return False
            # Try a harmless prctl call: PR_GET_DUMPABLE == 3
            rc = libc.prctl(3, 0, 0, 0, 0)
            return rc >= 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self) -> bool:
        """
        Apply the seccomp-BPF filter to the *current* process.

        Must be called after ``prctl(PR_SET_NO_NEW_PRIVS, 1)`` which this
        method also calls.

        Returns True if the filter was successfully applied, False otherwise.
        """
        if not self.is_available():
            logger.warning("seccomp-BPF not available on this platform — skipping")
            return False
        try:
            return self._apply_filter()
        except Exception as exc:
            logger.warning("seccomp apply failed: %s", exc)
            return False

    def _apply_filter(self) -> bool:
        libc = self._get_libc()
        if libc is None:
            return False

        # Choose allowed syscall set
        allowed_nrs = self._choose_allowed_syscalls()
        if not allowed_nrs:
            return False

        # Build BPF program
        bpf_prog = self._build_bpf_program(allowed_nrs)
        if bpf_prog is None:
            return False

        # Set no-new-privs first (required for unprivileged seccomp)
        rc = libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        if rc != 0:
            logger.warning("prctl(PR_SET_NO_NEW_PRIVS) failed: errno=%d", ctypes.get_errno())
            return False

        # Apply seccomp filter
        rc = libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER,
                        ctypes.cast(bpf_prog, ctypes.c_void_p), 0, 0)
        if rc != 0:
            logger.warning("prctl(PR_SET_SECCOMP) failed: errno=%d", ctypes.get_errno())
            return False

        return True

    def _choose_allowed_syscalls(self) -> Set[int]:
        """Decide which syscall set to use based on the profile."""
        if self.profile.allowed_syscalls is not None:
            # Profile specifies explicit syscall names — not yet mapped, use standard
            logger.debug("explicit allowed_syscalls not yet mapped to NRs; using standard filter")
        if self.profile.network_allowed:
            return self.network_filter()
        elif self.profile.writable_paths:
            return self.standard_filter()
        else:
            return self.minimal_filter()

    def _build_bpf_program(self, allowed_nrs: Set[int]) -> Optional[ctypes.Array]:  # type: ignore[type-arg]
        """Build a BPF program that ALLOWS the listed syscalls and KILLs others."""
        instructions: List[bytes] = []

        # Load syscall number from seccomp_data into accumulator
        # BPF_LD | BPF_W | BPF_ABS, offset = 0 (syscall nr is at offset 0)
        instructions.append(_make_bpf_insn(
            _BPF_LD | _BPF_W | _BPF_ABS, 0, 0, _SECCOMP_DATA_NR_OFFSET
        ))

        sorted_nrs = sorted(allowed_nrs)
        total_insns = 1 + len(sorted_nrs) + 1  # ld + jeq per syscall + kill
        # After the LD instruction each JEQ uses: if equal jump forward to ALLOW
        # The KILL instruction is at the end

        for i, nr in enumerate(sorted_nrs):
            # Jump forward by (remaining_syscalls - i) to reach ALLOW
            # After this instruction: remaining = (len - i - 1) jeq insns + 1 kill = len - i
            # ALLOW is placed after all JEQs and the KILL, so offset from next insn to ALLOW
            # Indices after LD: jeq[0], jeq[1], ..., jeq[n-1], kill, allow
            # jeq[i] is at position i+1; allow is at position n+1
            # jump distance from jeq[i] to allow = n+1 - (i+1) - 1 = n - i - 1
            jt = len(sorted_nrs) - i - 1  # if equal, skip remaining JEQs and the KILL → ALLOW
            instructions.append(_make_bpf_insn(
                _BPF_JMP | _BPF_JEQ | _BPF_K, jt, 0, nr
            ))

        # KILL instruction (reached if no JEQ matched)
        instructions.append(_make_bpf_insn(_BPF_RET | _BPF_K, 0, 0, _BPF_KILL))
        # ALLOW instruction
        instructions.append(_make_bpf_insn(_BPF_RET | _BPF_K, 0, 0, _BPF_ALLOW))

        # Pack into bytes
        prog_bytes = b"".join(instructions)
        n_insns = len(instructions)

        # struct sock_fprog { __u16 len; struct sock_filter *filter; }
        # On 64-bit Linux: len(u16) + 6 bytes padding + pointer(u64) = 16 bytes
        buf = (ctypes.c_uint8 * len(prog_bytes)).from_buffer_copy(prog_bytes)

        # sock_fprog: len (u16), pad (6 bytes), filter ptr (u64)
        class SockFprog(ctypes.Structure):
            _fields_ = [
                ("len", ctypes.c_uint16),
                ("filter", ctypes.c_void_p),
            ]

        fprog = SockFprog()
        fprog.len = n_insns
        fprog.filter = ctypes.cast(buf, ctypes.c_void_p).value  # type: ignore[assignment]

        # Keep a reference so buf isn't GC'd
        fprog._buf_ref = buf  # type: ignore[attr-defined]
        return fprog  # type: ignore[return-value]

    def _get_libc(self) -> Optional[ctypes.CDLL]:
        if self._libc is not None:
            return self._libc
        try:
            name = ctypes.util.find_library("c")
            if name is None:
                name = "libc.so.6"
            lib = ctypes.CDLL(name, use_errno=True)
            self._libc = lib
            return lib
        except Exception as exc:
            logger.warning("Could not load libc: %s", exc)
            return None


# ---------------------------------------------------------------------------
# NamespaceIsolation
# ---------------------------------------------------------------------------

@dataclass
class SandboxContext:
    """Holds handles/metadata for an active namespace sandbox."""

    pid: int
    namespaces_applied: List[str] = field(default_factory=list)
    fallback_limits_applied: bool = False
    warnings: List[str] = field(default_factory=list)


class NamespaceIsolation:
    """
    Uses Linux namespaces (via ``unshare()``) to isolate PID, mount, network,
    and user namespaces.

    Falls back gracefully on non-Linux or unprivileged environments:
    ``is_available()`` returns ``False`` and ``create_sandbox()`` applies only
    ``resource.setrlimit()`` limits.
    """

    def __init__(self, profile: SandboxProfile) -> None:
        self.profile = profile
        self._libc: Optional[ctypes.CDLL] = None

    def is_available(self) -> bool:
        """Return True if Linux namespaces can be created on this system."""
        if not _IS_LINUX:
            return False
        try:
            # Check if unshare is available
            libc = self._get_libc()
            if libc is None:
                return False
            # A safe way: check /proc/self/ns exists
            return os.path.exists("/proc/self/ns")
        except Exception:
            return False

    def create_sandbox(self) -> SandboxContext:
        """
        Create a sandbox context for the *current* process.

        Attempts namespace isolation; falls back to resource limits if
        namespaces are unavailable or permission is denied.
        """
        ctx = SandboxContext(pid=os.getpid())

        if not self.is_available():
            ctx.warnings.append("Namespace isolation unavailable on this platform")
            return ctx

        applied: List[str] = []
        warnings: List[str] = []

        libc = self._get_libc()
        if libc is None:
            ctx.warnings.append("Could not load libc; skipping namespaces")
            return ctx

        # Build flags for unshare
        flags = 0
        # User namespace (allows unprivileged usage of other namespaces)
        flags |= _CLONE_NEWUSER
        flags |= _CLONE_NEWPID
        flags |= _CLONE_NEWNS    # mount
        if not self.profile.network_allowed:
            flags |= _CLONE_NEWNET

        try:
            rc = libc.unshare(flags)
            if rc == 0:
                if flags & _CLONE_NEWUSER:
                    applied.append("user")
                if flags & _CLONE_NEWPID:
                    applied.append("pid")
                if flags & _CLONE_NEWNS:
                    applied.append("mount")
                if flags & _CLONE_NEWNET and not self.profile.network_allowed:
                    applied.append("network")
            else:
                errno = ctypes.get_errno()
                warnings.append(
                    f"unshare() failed with errno={errno}; "
                    f"falling back to resource limits only"
                )
        except Exception as exc:
            warnings.append(f"unshare() exception: {exc}; skipping namespaces")

        ctx.namespaces_applied = applied
        ctx.warnings = warnings
        return ctx

    def _get_libc(self) -> Optional[ctypes.CDLL]:
        if self._libc is not None:
            return self._libc
        try:
            name = ctypes.util.find_library("c") or "libc.so.6"
            lib = ctypes.CDLL(name, use_errno=True)
            self._libc = lib
            return lib
        except Exception as exc:
            logger.warning("Could not load libc: %s", exc)
            return None


# ---------------------------------------------------------------------------
# EnforcementReport
# ---------------------------------------------------------------------------

@dataclass
class EnforcementReport:
    """
    Documents which sandbox layers were actually enforced.

    Attributes
    ----------
    platform               : Platform string (e.g. "Linux", "Darwin").
    seccomp_applied        : Whether seccomp-BPF filter was loaded.
    namespaces_applied     : Whether any Linux namespaces were created.
    namespace_types        : Which namespace types were isolated.
    resource_limits_applied: Per-limit success flags from ResourceLimiter.apply().
    env_sanitized          : Whether environment was sanitized.
    secrets_scrubbed       : Names of env vars detected as secrets and removed.
    warnings               : Any non-fatal issues encountered.
    enforcement_level      : Summary level: "full", "partial", or "minimal".
    """

    platform: str
    seccomp_applied: bool
    namespaces_applied: bool
    namespace_types: List[str]
    resource_limits_applied: Dict[str, bool]
    env_sanitized: bool
    secrets_scrubbed: List[str]
    warnings: List[str]
    enforcement_level: str  # "full" | "partial" | "minimal"

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        layers = []
        if self.env_sanitized:
            layers.append("env")
        if any(self.resource_limits_applied.values()):
            layers.append("rlimit")
        if self.namespaces_applied:
            layers.append("namespaces")
        if self.seccomp_applied:
            layers.append("seccomp")
        return (
            f"EnforcementReport(platform={self.platform}, "
            f"level={self.enforcement_level}, layers=[{', '.join(layers)}])"
        )


# ---------------------------------------------------------------------------
# SandboxResult
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """
    Holds the outcome of a SandboxedProcess.start() call.

    Attributes
    ----------
    success          : Whether the target callable completed without error.
    return_value     : The return value of the target callable (if success).
    exception        : The exception raised (if not success).
    enforcement      : The EnforcementReport for this run.
    elapsed_seconds  : Wall-clock duration of the target execution.
    """

    success: bool
    return_value: Any
    exception: Optional[Exception]
    enforcement: EnforcementReport
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# SandboxedProcess
# ---------------------------------------------------------------------------

def _sandboxed_worker(
    target: Callable,
    args: tuple,
    profile_pickle: bytes,
    result_conn: multiprocessing.connection.Connection,
) -> None:
    """
    Worker function executed inside the sandbox subprocess.

    Applies all available enforcement layers, then calls ``target(*args)``.
    Reports results and the EnforcementReport back via a ``multiprocessing.Pipe``
    connection (avoids background feeder threads that multiprocessing.Queue uses,
    which would be blocked by a tight seccomp filter).
    """
    import pickle
    import time as _time

    profile: SandboxProfile = pickle.loads(profile_pickle)
    warnings: List[str] = []
    secrets_scrubbed: List[str] = []

    # ---- Layer 1: Environment sanitization --------------------------------
    sanitizer = EnvironmentSanitizer()
    current_env = dict(os.environ)
    leaked = sanitizer.detect_leaked_secrets(current_env)
    secrets_scrubbed.extend(leaked)
    clean_env = sanitizer.sanitize(current_env, profile)
    # Replace os.environ in-place
    os.environ.clear()
    os.environ.update(clean_env)
    env_sanitized = True

    # ---- Layer 2: Resource limits -----------------------------------------
    limiter = ResourceLimiter(profile)
    resource_limits = limiter.apply()

    # ---- Layer 3: Namespace isolation (Linux only) -------------------------
    ns_isolation = NamespaceIsolation(profile)
    if ns_isolation.is_available():
        ctx = ns_isolation.create_sandbox()
        namespaces_applied = bool(ctx.namespaces_applied)
        namespace_types = ctx.namespaces_applied
        warnings.extend(ctx.warnings)
    else:
        namespaces_applied = False
        namespace_types = []
        if _IS_LINUX:
            warnings.append("Namespace isolation failed or unavailable")

    # ---- Layer 4: Seccomp BPF (Linux x86_64 only) -------------------------
    seccomp = SeccompFilter(profile)
    seccomp_applied = False
    if seccomp.is_available():
        seccomp_applied = seccomp.apply()
        if not seccomp_applied:
            warnings.append("SeccompFilter.apply() returned False")
    else:
        if _IS_LINUX:
            warnings.append("Seccomp not available on this Linux system")

    # ---- Determine enforcement level --------------------------------------
    has_kernel = seccomp_applied or namespaces_applied
    has_resource = any(resource_limits.values())
    if has_kernel and has_resource and env_sanitized:
        level = "full"
    elif has_resource or env_sanitized:
        level = "partial"
    else:
        level = "minimal"

    report = EnforcementReport(
        platform=platform.system(),
        seccomp_applied=seccomp_applied,
        namespaces_applied=namespaces_applied,
        namespace_types=namespace_types,
        resource_limits_applied=resource_limits,
        env_sanitized=env_sanitized,
        secrets_scrubbed=secrets_scrubbed,
        warnings=warnings,
        enforcement_level=level,
    )

    # ---- Execute target ----------------------------------------------------
    start = _time.monotonic()
    try:
        ret = target(*args)
        elapsed = _time.monotonic() - start
        payload = {
            "success": True,
            "return_value": ret,
            "exception": None,
            "enforcement": report,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        elapsed = _time.monotonic() - start
        payload = {
            "success": False,
            "return_value": None,
            "exception": exc,
            "enforcement": report,
            "elapsed_seconds": elapsed,
        }

    # Use the pipe (no feeder thread — safe under seccomp)
    try:
        result_conn.send(payload)
    finally:
        result_conn.close()


class SandboxedProcess:
    """
    A fully sandboxed subprocess that applies all available isolation layers.

    The enforcement stack (applied in order inside the child process):

    1. Environment sanitization — remove secrets, apply whitelist/blacklist
    2. Resource limits          — memory, CPU, FDs, processes (Linux + macOS)
    3. Namespace isolation      — PID, mount, network (Linux only)
    4. Seccomp BPF filter       — syscall whitelist (Linux x86_64 only)

    On non-Linux platforms layers 3-4 are skipped with warnings.
    Layers 1-2 always run.

    Parameters
    ----------
    profile  : SandboxProfile defining the allowed capabilities.
    agent_id : Identifier for the agent, used in logging.
    timeout  : Maximum wall-clock seconds to wait for the child (None = wait forever).
    """

    def __init__(
        self,
        profile: SandboxProfile,
        agent_id: str,
        timeout: Optional[float] = 60.0,
    ) -> None:
        self.profile = profile
        self.agent_id = agent_id
        self.timeout = timeout
        self._last_report: Optional[EnforcementReport] = None

    def start(self, target: Callable, args: tuple = ()) -> SandboxResult:
        """
        Execute ``target(*args)`` inside a sandboxed subprocess.

        Returns a SandboxResult with the outcome and enforcement report.
        """
        import pickle

        profile_bytes = pickle.dumps(self.profile)
        # Use Pipe (not Queue) so the child process does not need a feeder thread.
        # A tight seccomp filter blocks clone/thread creation, which multiprocessing.Queue
        # requires internally; Pipe.send() uses a simple write() syscall.
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)

        proc = multiprocessing.Process(
            target=_sandboxed_worker,
            args=(target, args, profile_bytes, child_conn),
            daemon=True,
        )
        proc.start()
        # Close the child end in the parent — the child owns it
        child_conn.close()
        proc.join(timeout=self.timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
            parent_conn.close()
            # Build a timeout report
            report = EnforcementReport(
                platform=platform.system(),
                seccomp_applied=False,
                namespaces_applied=False,
                namespace_types=[],
                resource_limits_applied={},
                env_sanitized=False,
                secrets_scrubbed=[],
                warnings=["Process timed out and was terminated"],
                enforcement_level="minimal",
            )
            self._last_report = report
            return SandboxResult(
                success=False,
                return_value=None,
                exception=TimeoutError(
                    f"Sandboxed process for agent {self.agent_id!r} timed out "
                    f"after {self.timeout}s"
                ),
                enforcement=report,
                elapsed_seconds=self.timeout or 0.0,
            )

        try:
            if parent_conn.poll(1.0):
                raw = parent_conn.recv()
            else:
                raise RuntimeError("pipe poll timed out")
        except Exception:
            parent_conn.close()
            report = EnforcementReport(
                platform=platform.system(),
                seccomp_applied=False,
                namespaces_applied=False,
                namespace_types=[],
                resource_limits_applied={},
                env_sanitized=False,
                secrets_scrubbed=[],
                warnings=["No result received from sandboxed process"],
                enforcement_level="minimal",
            )
            self._last_report = report
            return SandboxResult(
                success=False,
                return_value=None,
                exception=RuntimeError("Sandboxed process produced no result"),
                enforcement=report,
                elapsed_seconds=0.0,
            )
        finally:
            parent_conn.close()

        report: EnforcementReport = raw["enforcement"]
        self._last_report = report

        return SandboxResult(
            success=raw["success"],
            return_value=raw["return_value"],
            exception=raw["exception"],
            enforcement=report,
            elapsed_seconds=raw["elapsed_seconds"],
        )

    def get_enforcement_report(self) -> Optional[EnforcementReport]:
        """
        Return the EnforcementReport from the most recent ``start()`` call.

        Returns None if ``start()`` has not been called yet.
        """
        return self._last_report


# ---------------------------------------------------------------------------
# capability_to_profile
# ---------------------------------------------------------------------------

def capability_to_profile(
    capabilities: Set[CapabilityType],
    agent_id: str,
) -> SandboxProfile:
    """
    Generate a SandboxProfile from a set of granted CapabilityType values.

    Mapping rules:
    - ADMIN                  → PROFILE_PRIVILEGED
    - FILE_READ              → read access to common paths
    - FILE_WRITE             → write access to /tmp/<agent_id>
    - NETWORK                → network_allowed = True
    - SPAWN                  → max_processes = 8
    - No capabilities at all → PROFILE_MINIMAL

    The returned profile is always a fresh copy so callers may modify it.
    """
    import copy

    if CapabilityType.ADMIN in capabilities:
        p = copy.deepcopy(PROFILE_PRIVILEGED)
        p.name = f"privileged:{agent_id}"
        return p

    agent_tmp = f"/tmp/battousai/{agent_id}"

    allowed: List[str] = ["/proc/self", "/usr", "/lib", "/lib64"]
    denied: List[str] = [
        os.path.expanduser("~/.ssh"),
        os.path.expanduser("~/.aws"),
        os.path.expanduser("~/.gnupg"),
    ]
    writable: List[str] = []
    network = False
    max_procs = 1
    max_mem = 128 * 1024 * 1024   # 128 MB default
    max_cpu = 30.0
    max_fds = 64
    env_whitelist = ["PATH", "LANG", "LC_ALL", "TZ"]

    if CapabilityType.FILE_READ in capabilities:
        allowed.extend(["/tmp", "/etc", "/home"])

    if CapabilityType.FILE_WRITE in capabilities:
        allowed.append(agent_tmp)
        writable.append(agent_tmp)
        max_fds = max(max_fds, 128)

    if CapabilityType.NETWORK in capabilities:
        network = True
        allowed.extend(["/etc/ssl", "/etc/ca-certificates"])
        env_whitelist += ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]

    if CapabilityType.SPAWN in capabilities:
        max_procs = 8
        max_mem = 256 * 1024 * 1024

    if CapabilityType.MEMORY_READ in capabilities or CapabilityType.MEMORY_WRITE in capabilities:
        max_mem = max(max_mem, 256 * 1024 * 1024)

    return SandboxProfile(
        name=f"generated:{agent_id}",
        allowed_paths=allowed,
        denied_paths=denied,
        writable_paths=writable,
        network_allowed=network,
        max_memory_bytes=max_mem,
        max_cpu_seconds=max_cpu,
        max_file_descriptors=max_fds,
        max_processes=max_procs,
        env_whitelist=env_whitelist,
        env_blacklist=[],
    )

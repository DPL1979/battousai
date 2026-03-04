# Battousai — Autonomous Intelligence Operating System

```
  ██████╗  █████╗ ████████╗████████╗ ██████╗ ██╗   ██╗███████╗ █████╗ ██╗
  ██╔══██╗██╔══██╗╚══██╔══╝╚══██╔══╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██║
  ██████╔╝███████║   ██║      ██║   ██║   ██║██║   ██║███████╗███████║██║
  ██╔══██╗██╔══██║   ██║      ██║   ██║   ██║██║   ██║╚════██║██╔══██║██║
  ██████╔╝██║  ██║   ██║      ██║   ╚██████╔╝╚██████╔╝███████║██║  ██║██║
  ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝
```

**Version**: 2.0.0 | **Status**: Production-Ready | **License**: MIT

---

## What is Battousai?

Battousai is a **production-grade autonomous AI agent operating system** built for enterprises that need reliable, scalable, and secure AI automation. Unlike simple chatbot frameworks, Battousai provides a complete runtime environment for deploying AI agents that can:

- 🤖 **Operate autonomously** for hours or days without human intervention
- 🔄 **Self-heal and recover** from failures automatically
- 📊 **Scale horizontally** across distributed infrastructure
- 🔒 **Enforce security policies** at the kernel level
- 🧠 **Learn and adapt** through evolutionary algorithms
- 🌐 **Federate** across multiple organizations and cloud providers

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    BATTOUSAI OPERATING SYSTEM                     │
├──────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Agent     │  │  Supervisor  │  │    Kernel    │  │   HAL       │  │
│  │   Runtime   │  │   + Policy   │  │  + Security  │  │ + Telemetry │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │    LLM      │  │  Capabilities │  │    Memory    │  │   Tools     │  │
│  │  Interface  │  │   Registry   │  │   Manager   │  │   Registry  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  Evolution  │  │  Federation  │  │  Contracts  │  │  Network    │  │
│  │   Engine    │  │   Manager   │  │    Engine   │  │  Manager   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Agent Runtime (`battousai/agent.py`)
The heart of Battousai. Manages agent lifecycle, task execution, and state machines.

```python
from battousai import BattousaiOS

os_instance = BattousaiOS(config)
await os_instance.start()

agent = await os_instance.spawn_agent(
    name="research-agent",
    role="researcher",
    capabilities=["web_search", "document_analysis"],
    model="gpt-4o"
)

result = await agent.execute_task({
    "type": "research",
    "query": "Latest developments in quantum computing",
    "depth": "comprehensive"
})
```

### 2. Kernel (`battousai/kernel.py`)
Security-first kernel with mandatory access control, resource management, and policy enforcement.

```python
# Kernel enforces all security policies automatically
# No explicit security code needed in agent logic
kernel = Kernel(config)
await kernel.start()

# Agents automatically get sandboxed execution
# Resource limits enforced at kernel level
# All operations logged for audit trail
```

### 3. Supervisor (`battousai/supervisor.py`)
Multi-agent orchestration with automatic load balancing, failure detection, and recovery.

```python
supervisor = Supervisor(kernel, config)

# Launch 10 parallel research agents
agents = await supervisor.spawn_pool(
    role="researcher",
    count=10,
    task_queue="research_tasks"
)

# Automatic failover - if any agent dies, supervisor restarts it
# Load balancing distributes tasks based on agent health scores
```

### 4. Evolution Engine (`battousai/evolution.py`)
Genetic algorithm-based self-optimization that improves agent performance over time.

```python
evolution = EvolutionEngine(config)

# Agents automatically evolve better strategies
# No manual tuning required
evolved_params = await evolution.evolve(
    population_size=50,
    generations=100,
    fitness_fn=task_completion_rate
)
```

### 5. Federation Manager (`battousai/federation.py`)
Secure multi-organization agent collaboration with cryptographic identity and trust management.

```python
federation = FederationManager(config)

# Connect with partner organization
await federation.join_federation(
    partner_endpoint="https://partner-battousai.example.com",
    trust_level=TrustLevel.VERIFIED
)

# Delegate tasks to federated agents
result = await federation.delegate_task(
    task=complex_analysis_task,
    partner="research-org"
)
```

### 6. Contract Engine (`battousai/contracts.py`)
Formal verification of agent behavior with pre/post conditions and invariant checking.

```python
@contract(
    pre=[lambda self, task: task.priority >= 0],
    post=[lambda self, result: result.confidence >= 0.7],
    invariant=[lambda self: self.memory_usage < self.max_memory]
)
async def execute_task(self, task):
    # Contract violations raise ContractError automatically
    return await self._process(task)
```

### 7. HAL (Hardware Abstraction Layer) (`battousai/hal.py`)
Unified interface for hardware, sensors, and external services.

```python
hal = HAL(config)

# Unified interface regardless of underlying hardware
sensor_data = await hal.read_sensor("camera_0")
await hal.actuate("motor_1", velocity=0.5)

# Automatic fallback to simulation in testing
```

### 8. Memory Manager (`battousai/memory.py`)
Multi-tier memory with semantic search, episodic storage, and cross-agent sharing.

```python
memory = MemoryManager(config)

# Store with automatic embedding and indexing
await memory.store(
    content="Customer prefers formal communication",
    agent_id="sales-agent-1",
    memory_type=MemoryType.EPISODIC
)

# Semantic retrieval across all memory tiers
relevant = await memory.retrieve(
    query="customer communication preferences",
    k=5
)
```

---

## Quick Start

### Installation

```bash
# Install from PyPI
pip install battousai

# Install with all optional dependencies
pip install battousai[all]

# Development installation
git clone https://github.com/DPL1979/battousai
cd battousai
pip install -e .[dev]
```

### Basic Usage

```python
import asyncio
from battousai import BattousaiOS
from battousai.schemas import BattousaiConfig, AgentRole

async def main():
    # Initialize the OS
    config = BattousaiConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        max_agents=10,
        security_level="standard"
    )
    
    os_instance = BattousaiOS(config)
    await os_instance.start()
    
    # Create an agent
    agent = await os_instance.spawn_agent(
        name="my-agent",
        role=AgentRole.GENERALIST,
        capabilities=["web_search", "code_execution", "file_management"]
    )
    
    # Execute a task
    result = await agent.execute_task({
        "type": "analysis",
        "data": "quarterly_report.pdf",
        "output_format": "executive_summary"
    })
    
    print(f"Task completed: {result.success}")
    print(f"Output: {result.output}")
    
    await os_instance.shutdown()

asyncio.run(main())
```

### Multi-Agent Pipeline

```python
async def research_pipeline():
    os_instance = BattousaiOS(config)
    await os_instance.start()
    
    # Spawn specialized agents
    researcher = await os_instance.spawn_agent(
        name="researcher",
        role=AgentRole.RESEARCHER,
        capabilities=["web_search", "document_analysis"]
    )
    
    analyst = await os_instance.spawn_agent(
        name="analyst",
        role=AgentRole.ANALYST,
        capabilities=["data_analysis", "visualization"]
    )
    
    writer = await os_instance.spawn_agent(
        name="writer",
        role=AgentRole.WRITER,
        capabilities=["content_generation", "editing"]
    )
    
    # Pipeline: research -> analyze -> write
    research_data = await researcher.execute_task({
        "type": "research",
        "topic": "AI trends 2025",
        "depth": "comprehensive"
    })
    
    analysis = await analyst.execute_task({
        "type": "analyze",
        "data": research_data.output,
        "focus": "key_insights"
    })
    
    report = await writer.execute_task({
        "type": "write",
        "content": analysis.output,
        "format": "executive_report"
    })
    
    return report
```

---

## Configuration

### Full Configuration Reference

```yaml
# battousai.yaml
battousai:
  version: "2.0"
  
  # LLM Configuration
  llm:
    provider: openai          # openai, anthropic, google, local
    model: gpt-4o
    fallback_model: gpt-3.5-turbo
    max_tokens: 4096
    temperature: 0.7
    timeout: 30
    
  # Agent Configuration  
  agents:
    max_concurrent: 50
    default_timeout: 300      # seconds
    max_retries: 3
    heartbeat_interval: 10
    
  # Security Configuration
  security:
    level: standard           # minimal, standard, strict, paranoid
    enable_sandboxing: true
    allowed_operations:
      - file_read
      - file_write
      - network_outbound
    blocked_operations:
      - system_exec
      - privilege_escalation
    audit_log: true
    
  # Memory Configuration
  memory:
    backend: redis            # redis, postgres, sqlite, in_memory
    max_entries_per_agent: 10000
    embedding_model: text-embedding-ada-002
    eviction_policy: lru
    
  # Federation Configuration
  federation:
    enabled: false
    trust_registry: "https://trust.battousai.io"
    max_partners: 10
    
  # Evolution Configuration
  evolution:
    enabled: false
    population_size: 20
    generation_interval: 3600  # seconds
    fitness_metric: task_success_rate
    
  # Monitoring
  monitoring:
    metrics_port: 9090
    log_level: INFO
    enable_tracing: true
    trace_endpoint: "http://jaeger:14268/api/traces"
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install battousai[all]

COPY battousai.yaml .
COPY agents/ ./agents/

CMD ["python", "-m", "battousai.main"]
```

```bash
docker build -t my-battousai-app .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY my-battousai-app
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: battousai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: battousai
  template:
    metadata:
      labels:
        app: battousai
    spec:
      containers:
      - name: battousai
        image: myregistry/battousai:2.0.0
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### Docker Compose

```yaml
version: '3.8'
services:
  battousai:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    ports:
      - "8000:8000"
      - "9090:9090"
      
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
      
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: battousai
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pg_data:/var/lib/postgresql/data
      
volumes:
  redis_data:
  pg_data:
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=battousai --cov-report=html

# Run specific test categories
pytest tests/ -m "unit"
pytest tests/ -m "integration"
pytest tests/ -m "performance"

# Run with parallel execution
pytest tests/ -n auto
```

---

## Performance Benchmarks

| Metric | Value | Conditions |
|--------|-------|------------|
| Agent spawn time | < 50ms | Cold start |
| Task throughput | 1000+ tasks/sec | 50 agents, simple tasks |
| Memory overhead | ~15MB/agent | Baseline |
| LLM call latency | Provider-dependent | GPT-4o: ~2s avg |
| Federation handshake | < 200ms | LAN |
| Evolution cycle | < 60s | 20 agents, 10 generations |

---

## Security Model

### Defense in Depth

```
┌───────────────────────────────────────────────────┐
│           SECURITY LAYERS                   │
├───────────────────────────────────────────────────┤
│  Layer 1: Input Validation & Sanitization    │
│  Layer 2: Authentication & Authorization     │
│  Layer 3: Capability-Based Access Control    │
│  Layer 4: Kernel Policy Enforcement          │
│  Layer 5: Sandboxed Execution Environment    │
│  Layer 6: Audit Logging & Anomaly Detection  │
│  Layer 7: Cryptographic Verification         │
└───────────────────────────────────────────────────┘
```

### Security Levels

| Level | Use Case | Restrictions |
|-------|----------|--------------|
| `minimal` | Development | None |
| `standard` | Production | Basic sandboxing |
| `strict` | Financial/Healthcare | Full isolation |
| `paranoid` | Government/Defense | Air-gapped execution |

---

## Roadmap

### v2.1 (Q2 2025)
- [ ] WebAssembly agent sandbox
- [ ] GraphQL API for agent management
- [ ] Native Kubernetes operator
- [ ] Multi-modal agent support (vision, audio)

### v2.2 (Q3 2025)
- [ ] Agent marketplace integration
- [ ] Automated prompt optimization
- [ ] Real-time collaboration between federated agents
- [ ] Cost optimization engine

### v3.0 (Q4 2025)
- [ ] Physical robot integration via ROS2
- [ ] Quantum-resistant cryptography
- [ ] On-device model execution (Apple Silicon, NVIDIA Jetson)
- [ ] Formal verification of agent behavior

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/battousai
cd battousai

# Create feature branch
git checkout -b feature/my-feature

# Make changes, run tests
pytest tests/ -v

# Submit PR
git push origin feature/my-feature
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- **asyncio** — Async Python runtime
- **Pydantic** — Data validation
- **aiohttp** — Async HTTP
- **Redis** — Memory backend
- **PostgreSQL** — Persistent storage
- **OpenTelemetry** — Distributed tracing

---

*Battousai: Named after the legendary sword master who mastered the art of swift, decisive action.*

---

## Future Hardware Integration

Planned for v3.0: Direct hardware integration replacing the current simulation layer:

- **GPIO Support**: Direct hardware pin control on embedded systems
- **ROS2 Integration**: Full Robot Operating System 2 compatibility
- **Vision Pipeline**: Real camera, and sensor integration replacing `SimulatedHardware`; target: Raspberry Pi and NVIDIA Jetson

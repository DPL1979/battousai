# Hardware Abstraction Layer

The `hal.py` module provides a unified interface for agents to interact with physical hardware. All devices are simulated in the prototype with realistic synthetic data, enabling agents to be developed and tested without real hardware.

---

## Design Rationale

Agents never access hardware directly. They go through:

```
agent.use_tool("hw_read", device_id="sensor_temp_0")
    │
    ▼
ToolManager.execute("hw_read", ...)
    │
    ▼
DeviceManager.read(device_id="sensor_temp_0")
    │
    ▼
SensorDevice.read() → {"reading": 23.4, "unit": "celsius", "timestamp": tick}
```

This architecture means an agent written for simulation works identically with real hardware — only the `DeviceManager` configuration changes.

---

## `DeviceType` Enum

```python
class DeviceType(Enum):
    GPIO       = auto()  # Digital general-purpose I/O (LEDs, buttons, relays)
    SENSOR     = auto()  # Analog/digital sensing
    CAMERA     = auto()  # Image capture
    ACTUATOR   = auto()  # Motors, servos, linear actuators
    DISPLAY    = auto()  # Output screens, LED matrices
    NETWORK_HW = auto()  # Physical network interfaces (WiFi, Bluetooth, LoRa)
    STORAGE    = auto()  # Physical storage media
    COMPUTE    = auto()  # Hardware accelerators (GPU, TPU, FPGA)
```

---

## `DeviceState` Enum

```python
class DeviceState(Enum):
    UNINITIALIZED = auto()  # Not yet set up
    READY         = auto()  # Available for use
    BUSY          = auto()  # Performing a long operation
    ERROR         = auto()  # Hardware fault; needs reset()
    OFFLINE       = auto()  # Device physically disconnected
```

Transitions: `UNINITIALIZED → READY` (via `initialize()`), `READY ↔ BUSY`, `READY/BUSY → ERROR`, `ERROR → READY` (via `reset()`), `any → OFFLINE`.

---

## `HardwareDevice` ABC

Base class for all Battousai hardware devices:

```python
from abc import ABC, abstractmethod

class HardwareDevice(ABC):
    device_id: str
    device_type: DeviceType
    state: DeviceState
    metadata: Dict[str, Any]

    @abstractmethod
    def initialize(self) -> bool: ...

    @abstractmethod
    def read(self) -> Dict[str, Any]: ...

    @abstractmethod
    def write(self, value: Any) -> bool: ...

    def reset(self) -> bool: ...
    def get_info(self) -> Dict[str, Any]: ...
```

---

## Device Types

### GPIO (Digital I/O)

State machine with digital read/write. Simulates LEDs, buttons, relays, and logic pins.

```python
# Read a GPIO pin state
r = self.use_tool("hw_read", device_id="gpio_led_0")
# {"state": 0, "device_id": "gpio_led_0", "type": "GPIO", "timestamp": tick}

# Write a GPIO pin (set LED on)
r = self.use_tool("hw_write", device_id="gpio_led_0", value=1)
# {"success": True, "previous_state": 0, "new_state": 1}
```

### Sensor

Analog sensor with realistic diurnal cycles and Gaussian noise:

- **Temperature:** sine-wave 24-hour cycle, σ=0.5°C noise
- **Light:** brightness cycle peaking at simulated noon
- **Distance:** ultrasonic-style ranging with noise
- **IMU:** accelerometer/gyroscope with drift simulation

```python
r = self.use_tool("hw_read", device_id="sensor_temp_0")
# {"reading": 23.4, "unit": "celsius", "device_id": "sensor_temp_0",
#  "type": "SENSOR", "sensor_type": "temperature", "timestamp": tick}

r = self.use_tool("hw_read", device_id="sensor_light_0")
# {"reading": 847.2, "unit": "lux", ...}
```

### Camera

Returns deterministic base64-encoded synthetic frames:

```python
r = self.use_tool("hw_read", device_id="camera_0")
# {"frame": "<base64 encoded PNG>", "width": 640, "height": 480,
#  "format": "RGB", "frame_number": 42, "timestamp": tick}
```

### Actuator

Tracks position, velocity, and target for motors/servos:

```python
# Command actuator to target position
r = self.use_tool("hw_write", device_id="servo_0", value={"target": 90.0, "speed": 0.5})
# {"success": True, "target": 90.0, "current_position": 45.2, "moving": True}

# Read current position
r = self.use_tool("hw_read", device_id="servo_0")
# {"position": 87.3, "velocity": 5.2, "target": 90.0, "at_target": False, "unit": "degrees"}
```

### Compute Accelerator

Simulates workload queuing for GPU/TPU/FPGA:

```python
# Submit a workload
r = self.use_tool("hw_write", device_id="gpu_0", value={
    "operation": "matrix_multiply",
    "input_shape": [1024, 1024],
    "dtype": "float32",
})
# {"success": True, "job_id": "job_0042", "estimated_ticks": 3}

# Check status
r = self.use_tool("hw_read", device_id="gpu_0")
# {"state": "BUSY", "queue_depth": 1, "jobs_completed": 41, "utilization": 0.85}
```

---

## `SimulatedHardware`

Factory for adding simulated devices:

```python
from battousai.hal import SimulatedHardware, DeviceType

hw = SimulatedHardware()

# Add devices
hw.add_sensor(
    device_id="temp_0",
    device_type=DeviceType.SENSOR,
    sensor_type="temperature",
    baseline=20.0,    # baseline temperature °C
    amplitude=5.0,    # diurnal swing ±5°C
    noise_std=0.5,    # Gaussian noise σ=0.5
)

hw.add_gpio(
    device_id="led_0",
    device_type=DeviceType.GPIO,
    initial_state=0,  # off
)

hw.add_camera(
    device_id="cam_0",
    device_type=DeviceType.CAMERA,
    width=640,
    height=480,
    frame_rate=30,
)

hw.add_actuator(
    device_id="motor_0",
    device_type=DeviceType.ACTUATOR,
    actuator_type="servo",
    initial_position=0.0,
    max_position=180.0,
    speed=10.0,  # degrees per tick
)

hw.add_compute(
    device_id="gpu_0",
    device_type=DeviceType.COMPUTE,
    accelerator_type="GPU",
    compute_capacity=100,    # arbitrary units
    memory_gb=8.0,
)
```

---

## `DeviceManager`

Registry for all hardware devices:

```python
from battousai.hal import DeviceManager

manager = DeviceManager()
manager.register_hardware(hw)

# Read a device
result = manager.read("sensor_temp_0", current_tick=tick)

# Write to a device
result = manager.write("gpio_led_0", value=1, current_tick=tick)

# List all registered devices
devices = manager.list_devices()
# [{"device_id": "temp_0", "type": "SENSOR", "state": "READY"}, ...]

# Get device info
info = manager.get_device_info("cam_0")

# Unregister when done
manager.unregister_device("temp_0")
```

---

## Registering HAL Tools with the Kernel

Expose hardware as kernel tools so agents can use them via `use_tool()`:

```python
from battousai.hal import SimulatedHardware, DeviceManager, DeviceType

# Create hardware
hw = SimulatedHardware()
hw.add_sensor("sensor_temp_0", DeviceType.SENSOR, sensor_type="temperature",
              baseline=20.0, amplitude=5.0)
hw.add_gpio("gpio_led_0", DeviceType.GPIO, initial_state=0)
hw.add_camera("camera_0", DeviceType.CAMERA, width=640, height=480)

manager = DeviceManager()
manager.register_hardware(hw)

# Register HAL tools with the kernel
# (hal.py provides a register_hal_tools function)
from battousai.hal import register_hal_tools

kernel.boot()
register_hal_tools(kernel.tools, manager)

# Now agents use: self.use_tool("hw_read", device_id="sensor_temp_0")
```

Registered tools:
- `hw_read` — read a device's current state
- `hw_write` — write a command or value to a device
- `hw_list` — list all registered devices

---

## Example: IoT Monitoring Agent

```python
from battousai.agent import Agent

class TemperatureMonitorAgent(Agent):
    """Reads temperature every tick and alerts on anomalies."""

    def __init__(self, alert_threshold: float = 30.0):
        super().__init__(name="TempMonitor", priority=5, memory_allocation=256)
        self._threshold = alert_threshold
        self._readings = []

    def on_spawn(self) -> None:
        self.log(f"Temperature monitor started. Alert at {self._threshold}°C")

    def think(self, tick: int) -> None:
        # Read temperature sensor
        r = self.use_tool("hw_read", device_id="sensor_temp_0")
        if r.ok:
            reading = r.value.get("reading", 0.0)
            self._readings.append(reading)

            # Store latest reading in memory
            self.mem_write("last_temp", reading)

            # Alert if threshold exceeded
            if reading > self._threshold:
                self.log(f"ALERT: Temperature {reading:.1f}°C exceeds {self._threshold}°C!")
                # Turn on warning LED
                self.use_tool("hw_write", device_id="gpio_led_0", value=1)
            else:
                # Turn off LED
                self.use_tool("hw_write", device_id="gpio_led_0", value=0)

            # Log periodically
            if tick % 5 == 0:
                avg = sum(self._readings[-10:]) / min(len(self._readings), 10)
                self.log(f"Tick {tick}: temp={reading:.1f}°C, 10-tick avg={avg:.1f}°C")

        self.yield_cpu()
```

---

## Related Pages

- [Architecture Overview](../architecture/overview.md) — HAL in the hardware layer
- [Built-in Tools](../tools/builtin.md) — tool registration pattern
- [Custom Agents](../agents/custom.md) — agents that use hardware tools

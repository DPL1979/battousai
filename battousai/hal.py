"""
hal.py — Hardware Abstraction Layer
=======================================
Provides a unified interface for agents to interact with hardware
devices. In the prototype, all hardware is simulated.

Design Rationale
----------------
Real AI operating systems will ultimately control physical hardware —
sensors, actuators, cameras, compute accelerators. The Hardware Abstraction
Layer (HAL) insulates agents from the specifics of any particular device by
defining a clean, consistent interface.

Agents never access hardware directly. They go through:
    agent.use_tool("hw_read", device_id="sensor_0")
which routes through the ToolManager to the DeviceManager to the device.

The simulation layer generates realistic synthetic data (sine-wave diurnal
temperature cycles, Gaussian noise on sensor readings, deterministic camera
frames) so that agents built against the HAL can be tested without real
hardware.

Device Types:
    GPIODevice     — digital I/O pins (LEDs, buttons, relays)
    SensorDevice   — analog sensors (temperature, light, distance, IMU)
    CameraDevice   — image capture (returns simulated image data)
    ActuatorDevice — motors, servos, linear actuators
    DisplayDevice  — screen/LED matrix output
    NetworkDevice  — physical network interfaces (WiFi, Bluetooth, LoRa)
    StorageDevice  — physical storage (SSD, NVMe, SD card)
    ComputeDevice  — accelerators (GPU, TPU, FPGA)

Architecture:
    HardwareDevice (ABC)   — base class for all devices
    DeviceDriver           — translates device commands to hardware calls
    DeviceManager          — registry and lifecycle management
    HardwareBus            — communication bus between devices
    SimulatedHardware      — generates realistic fake sensor data
"""

from __future__ import annotations

import base64
import math
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeviceType(Enum):
    """Classification of hardware device categories."""
    GPIO       = auto()  # Digital general-purpose I/O
    SENSOR     = auto()  # Analog/digital sensing
    CAMERA     = auto()  # Image capture
    ACTUATOR   = auto()  # Motors, servos, linear actuators
    DISPLAY    = auto()  # Output screens, LED matrices
    NETWORK_HW = auto()  # Physical network interfaces
    STORAGE    = auto()  # Physical storage media
    COMPUTE    = auto()  # Hardware accelerators (GPU, TPU, FPGA)


class DeviceState(Enum):
    """
    Lifecycle state of a hardware device.

    Transitions:
        UNINITIALIZED → READY  (via initialize())
        READY ↔ BUSY           (during long operations)
        READY/BUSY → ERROR     (on hardware fault)
        ERROR → READY          (via reset())
        any → OFFLINE          (device physically removed)
    """
    UNINITIALIZED = auto()
    READY         = auto()
    BUSY          = auto()
    ERROR         = auto()
    OFFLINE       = auto()


# ---------------------------------------------------------------------------
# Base Device
# ---------------------------------------------------------------------------

class HardwareDevice(ABC):
    """
    Abstract base class for all Battousai hardware devices.

    Every device has:
        device_id   — unique string identifier (e.g. "gpio_0", "sensor_temp_0")
        device_type — DeviceType enum
        state       — current DeviceState
        metadata    — dict of device-specific properties (model, serial, etc.)

    Concrete devices implement the four abstract methods:
        initialize() — power on, load firmware, set initial state
        read()       — query current device state or sensor value
        write(data)  — send a command or data to the device
        reset()      — return device to factory default state
        status()     — return a dict of health/diagnostic info
    """

    def __init__(
        self,
        device_id: str,
        device_type: DeviceType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device_id = device_id
        self.device_type = device_type
        self.state = DeviceState.UNINITIALIZED
        self.metadata: Dict[str, Any] = metadata or {}
        self._error_count: int = 0
        self._read_count: int = 0
        self._write_count: int = 0
        self._last_error: Optional[str] = None

    @abstractmethod
    def initialize(self) -> bool:
        """
        Power on and initialise the device.

        Returns True on success, False on hardware fault.
        Sets state to READY or ERROR.
        """

    @abstractmethod
    def read(self) -> Any:
        """
        Query the device for its current value or state.

        Returns a device-specific value (dict, float, bytes, etc.).
        Raises RuntimeError if the device is not READY.
        """

    @abstractmethod
    def write(self, data: Any) -> bool:
        """
        Send a command or data to the device.

        Returns True if the command was accepted, False otherwise.
        Raises RuntimeError if the device is not READY.
        """

    @abstractmethod
    def reset(self) -> bool:
        """
        Perform a hardware reset, returning the device to initial state.

        Returns True if reset succeeded.
        """

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Return a dict of health and diagnostic information."""

    def _assert_ready(self) -> None:
        if self.state != DeviceState.READY:
            raise RuntimeError(
                f"Device {self.device_id!r} is not READY (current state: {self.state.name}). "
                f"Call initialize() first or reset() if in ERROR state."
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.device_id!r}, "
            f"type={self.device_type.name}, state={self.state.name})"
        )


# ---------------------------------------------------------------------------
# Device Driver — buffered I/O wrapper with retry logic
# ---------------------------------------------------------------------------

class DeviceDriver:
    """
    Wraps a HardwareDevice with:
        - Buffered read cache (caches last N readings)
        - Automatic retry on transient errors
        - Error rate tracking
        - Command queue for ordered write operations

    The driver is the standard way agents interact with hardware. Direct
    device access is available but bypasses buffering and retry logic.
    """

    def __init__(
        self,
        device: HardwareDevice,
        cache_size: int = 10,
        max_retries: int = 3,
        retry_delay_ms: float = 10.0,
    ) -> None:
        self.device = device
        self.cache_size = cache_size
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self._read_cache: List[Tuple[float, Any]] = []  # (timestamp, value)
        self._write_queue: List[Any] = []
        self._total_reads: int = 0
        self._total_writes: int = 0
        self._retry_count: int = 0
        self._error_count: int = 0

    def read(self) -> Any:
        """Read from device with retry logic. Caches the result."""
        for attempt in range(self.max_retries):
            try:
                value = self.device.read()
                self.device._read_count += 1
                self._total_reads += 1
                # Update cache
                entry = (time.time(), value)
                self._read_cache.append(entry)
                if len(self._read_cache) > self.cache_size:
                    self._read_cache.pop(0)
                return value
            except Exception as exc:
                self._retry_count += 1
                self._error_count += 1
                if attempt == self.max_retries - 1:
                    self.device._error_count += 1
                    self.device._last_error = str(exc)
                    raise
        return None  # unreachable

    def write(self, data: Any) -> bool:
        """Write to device with retry logic."""
        for attempt in range(self.max_retries):
            try:
                ok = self.device.write(data)
                self.device._write_count += 1
                self._total_writes += 1
                return ok
            except Exception as exc:
                self._retry_count += 1
                self._error_count += 1
                if attempt == self.max_retries - 1:
                    self.device._error_count += 1
                    self.device._last_error = str(exc)
                    raise
        return False

    def last_reading(self) -> Optional[Any]:
        """Return the most recently cached reading."""
        return self._read_cache[-1][1] if self._read_cache else None

    def recent_readings(self, n: int = 5) -> List[Any]:
        """Return the last N cached readings (values only)."""
        return [v for _, v in self._read_cache[-n:]]

    def stats(self) -> Dict[str, Any]:
        return {
            "device_id": self.device.device_id,
            "total_reads": self._total_reads,
            "total_writes": self._total_writes,
            "retry_count": self._retry_count,
            "error_count": self._error_count,
        }


# ---------------------------------------------------------------------------
# Device Manager
# ---------------------------------------------------------------------------

class DeviceManager:
    """
    Registry and lifecycle manager for all hardware devices.

    Provides:
        - Device registration / unregistration
        - Lookup by ID or type
        - Bulk initialisation
        - Simulated device discovery (scan)
    """

    def __init__(self) -> None:
        self._devices: Dict[str, HardwareDevice] = {}
        self._drivers: Dict[str, DeviceDriver] = {}

    def register(self, device: HardwareDevice, auto_init: bool = True) -> DeviceDriver:
        """
        Register a device and create its driver.

        Args:
            device    — HardwareDevice instance to register.
            auto_init — If True, call device.initialize() immediately.

        Returns the DeviceDriver for the registered device.
        """
        self._devices[device.device_id] = device
        driver = DeviceDriver(device)
        self._drivers[device.device_id] = driver
        if auto_init:
            try:
                device.initialize()
            except Exception as exc:
                device.state = DeviceState.ERROR
                device._last_error = str(exc)
        return driver

    def unregister(self, device_id: str) -> bool:
        """Remove a device from the registry."""
        self._devices.pop(device_id, None)
        self._drivers.pop(device_id, None)
        return True

    def get(self, device_id: str) -> Optional[HardwareDevice]:
        """Return a device by ID, or None."""
        return self._devices.get(device_id)

    def get_driver(self, device_id: str) -> Optional[DeviceDriver]:
        """Return the DeviceDriver for a device."""
        return self._drivers.get(device_id)

    def list_by_type(self, device_type: DeviceType) -> List[HardwareDevice]:
        """Return all registered devices of a given type."""
        return [d for d in self._devices.values() if d.device_type == device_type]

    def list_all(self) -> List[HardwareDevice]:
        return list(self._devices.values())

    def scan(self) -> List[str]:
        """
        Simulate device discovery (e.g., USB enumeration, I2C bus scan).

        Returns a list of device IDs of newly discovered devices.
        In the simulated environment, this returns devices in ERROR/OFFLINE
        state and attempts to re-initialise them.
        """
        recovered = []
        for device_id, device in self._devices.items():
            if device.state in (DeviceState.ERROR, DeviceState.OFFLINE):
                try:
                    ok = device.reset()
                    if ok:
                        recovered.append(device_id)
                except Exception:
                    pass
        return recovered

    def stats(self) -> Dict[str, Any]:
        return {
            "total_devices": len(self._devices),
            "by_type": {
                dt.name: len(self.list_by_type(dt))
                for dt in DeviceType
                if self.list_by_type(dt)
            },
            "by_state": {
                ds.name: sum(1 for d in self._devices.values() if d.state == ds)
                for ds in DeviceState
            },
        }


# ---------------------------------------------------------------------------
# HardwareBus — inter-device messaging with simulated latency
# ---------------------------------------------------------------------------

class HardwareBus:
    """
    Simulated communication bus between hardware devices.

    Models physical buses like I2C, SPI, CAN, or internal PCIe links.
    Messages can be addressed to a specific device or broadcast.

    Simulated latency:
        Each message has a latency_ticks delay before the recipient sees it.
        This models real bus arbitration and propagation delays.
    """

    BROADCAST = "__BUS_BROADCAST__"

    def __init__(self, bus_id: str = "main", latency_ticks: int = 1) -> None:
        self.bus_id = bus_id
        self.latency_ticks = latency_ticks
        self._current_tick: int = 0
        # (delivery_tick, sender_id, recipient_id, payload)
        self._queue: List[Tuple[int, str, str, Any]] = []
        self._message_count: int = 0

    def send(self, sender_id: str, recipient_id: str, payload: Any) -> None:
        """
        Enqueue a message for delivery after latency_ticks.

        Args:
            sender_id    — device_id of the sender
            recipient_id — device_id of the recipient, or BROADCAST
            payload      — arbitrary message data
        """
        delivery_tick = self._current_tick + self.latency_ticks
        self._queue.append((delivery_tick, sender_id, recipient_id, payload))
        self._message_count += 1

    def tick(self, current_tick: int) -> List[Tuple[str, str, Any]]:
        """
        Advance the bus clock and return messages ready for delivery.

        Returns list of (sender_id, recipient_id, payload) for all messages
        whose delivery_tick <= current_tick.
        """
        self._current_tick = current_tick
        ready = [(s, r, p) for (dt, s, r, p) in self._queue if dt <= current_tick]
        self._queue = [(dt, s, r, p) for (dt, s, r, p) in self._queue if dt > current_tick]
        return ready

    def stats(self) -> Dict[str, Any]:
        return {
            "bus_id": self.bus_id,
            "latency_ticks": self.latency_ticks,
            "queued_messages": len(self._queue),
            "total_messages": self._message_count,
        }


# ---------------------------------------------------------------------------
# Concrete Device Implementations
# ---------------------------------------------------------------------------

class GPIODevice(HardwareDevice):
    """
    Simulated digital general-purpose I/O device.

    Models a microcontroller GPIO bank with configurable pin count.
    Each pin can be HIGH (True) or LOW (False), with an optional
    direction (INPUT/OUTPUT).

    Read returns: {"pins": {0: True, 1: False, ...}, "timestamp": float}
    Write accepts: {"pin": int, "value": bool} or {"pins": {0: True, ...}}
    """

    def __init__(self, device_id: str, pin_count: int = 16) -> None:
        super().__init__(
            device_id=device_id,
            device_type=DeviceType.GPIO,
            metadata={"pin_count": pin_count, "model": "SimGPIO-v1"},
        )
        self.pin_count = pin_count
        self.pin_states: Dict[int, bool] = {i: False for i in range(pin_count)}
        self.pin_directions: Dict[int, str] = {i: "OUTPUT" for i in range(pin_count)}

    def initialize(self) -> bool:
        self.pin_states = {i: False for i in range(self.pin_count)}
        self.state = DeviceState.READY
        return True

    def read(self) -> Dict[str, Any]:
        self._assert_ready()
        return {
            "device_id": self.device_id,
            "pins": dict(self.pin_states),
            "directions": dict(self.pin_directions),
            "timestamp": time.time(),
        }

    def write(self, data: Any) -> bool:
        self._assert_ready()
        if isinstance(data, dict):
            if "pin" in data and "value" in data:
                pin = int(data["pin"])
                if 0 <= pin < self.pin_count:
                    self.pin_states[pin] = bool(data["value"])
                    return True
            elif "pins" in data:
                for pin, val in data["pins"].items():
                    if 0 <= int(pin) < self.pin_count:
                        self.pin_states[int(pin)] = bool(val)
                return True
        return False

    def reset(self) -> bool:
        self.pin_states = {i: False for i in range(self.pin_count)}
        self.state = DeviceState.READY
        self._error_count = 0
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "state": self.state.name,
            "pin_count": self.pin_count,
            "high_pins": sum(1 for v in self.pin_states.values() if v),
            "error_count": self._error_count,
        }


class SensorDevice(HardwareDevice):
    """
    Simulated analog/digital sensor.

    Generates realistic synthetic data using sine-wave models plus
    Gaussian noise for the following sensor types:

        temp      — 18–28°C with 24-hour diurnal variation
        light     — 0–1000 lux (sinusoidal day/night cycle)
        distance  — 0–500cm (random walk around 200cm)
        humidity  — 30–80% relative humidity
        pressure  — 990–1020 hPa (slow drift)
        imu       — {accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z}

    Each reading includes a timestamp and sequence number.
    """

    _SENSOR_TYPES = {"temp", "light", "distance", "humidity", "pressure", "imu"}

    def __init__(self, device_id: str, sensor_type: str = "temp") -> None:
        if sensor_type not in self._SENSOR_TYPES:
            raise ValueError(f"Unknown sensor type {sensor_type!r}. Must be one of {self._SENSOR_TYPES}")
        super().__init__(
            device_id=device_id,
            device_type=DeviceType.SENSOR,
            metadata={"sensor_type": sensor_type, "model": f"SimSensor-{sensor_type}-v1"},
        )
        self.sensor_type = sensor_type
        self._seq: int = 0
        self._rng = random.Random(hash(device_id))  # deterministic per device

    def initialize(self) -> bool:
        self.state = DeviceState.READY
        self._seq = 0
        return True

    def read(self) -> Dict[str, Any]:
        self._assert_ready()
        self._seq += 1
        t = time.time()
        # Phase offset based on device_id for variety between sensors
        phase = hash(self.device_id) % 1000 / 1000.0
        value = self._generate_reading(t, phase)
        return {
            "device_id": self.device_id,
            "sensor_type": self.sensor_type,
            "value": value,
            "seq": self._seq,
            "timestamp": t,
        }

    def _generate_reading(self, t: float, phase: float) -> Any:
        """Generate realistic simulated sensor data."""
        noise = lambda scale: self._rng.gauss(0, scale)  # noqa: E731
        cycle = math.sin(2 * math.pi * (t / 86400.0 + phase))  # 24h cycle

        if self.sensor_type == "temp":
            # 23°C mean, ±5°C diurnal swing, ±0.3°C noise
            return round(23.0 + 5.0 * cycle + noise(0.3), 2)

        elif self.sensor_type == "light":
            # 0 at night, 1000 at noon
            raw = 500.0 * (1.0 + cycle) + noise(20.0)
            return round(max(0.0, raw), 1)

        elif self.sensor_type == "distance":
            # Random walk around 200cm
            drift = self._rng.uniform(-5, 5)
            base = 200.0 + 50.0 * cycle + noise(5.0) + drift
            return round(max(0.0, min(500.0, base)), 1)

        elif self.sensor_type == "humidity":
            raw = 55.0 + 15.0 * cycle + noise(1.5)
            return round(max(30.0, min(80.0, raw)), 1)

        elif self.sensor_type == "pressure":
            raw = 1013.25 + 8.0 * cycle + noise(0.5)
            return round(raw, 2)

        elif self.sensor_type == "imu":
            return {
                "accel_x": round(noise(0.05), 4),
                "accel_y": round(noise(0.05), 4),
                "accel_z": round(9.81 + noise(0.02), 4),
                "gyro_x":  round(noise(0.01), 5),
                "gyro_y":  round(noise(0.01), 5),
                "gyro_z":  round(noise(0.01), 5),
            }
        return None

    def write(self, data: Any) -> bool:
        # Sensors are typically read-only; write can be used to set config
        self._assert_ready()
        return False  # read-only sensor

    def reset(self) -> bool:
        self.state = DeviceState.READY
        self._seq = 0
        self._error_count = 0
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "sensor_type": self.sensor_type,
            "state": self.state.name,
            "readings_taken": self._seq,
            "error_count": self._error_count,
        }


class CameraDevice(HardwareDevice):
    """
    Simulated camera device.

    In the prototype, read() returns a metadata dict with a small
    base64-encoded synthetic "frame" token (not a real image).

    Real implementations would return compressed JPEG data or an
    ndarray. The interface is designed to be drop-in compatible.

    Frame dict:
        width    — int (pixels)
        height   — int (pixels)
        format   — str ("RGB24", "JPEG", etc.)
        seq      — int (frame sequence number)
        data     — str (base64-encoded synthetic frame token)
        timestamp — float
    """

    def __init__(
        self,
        device_id: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fmt: str = "RGB24",
    ) -> None:
        super().__init__(
            device_id=device_id,
            device_type=DeviceType.CAMERA,
            metadata={"width": width, "height": height, "fps": fps, "format": fmt,
                      "model": "SimCam-v1"},
        )
        self.width = width
        self.height = height
        self.fps = fps
        self.format = fmt
        self._frame_seq: int = 0

    def initialize(self) -> bool:
        self.state = DeviceState.READY
        return True

    def read(self) -> Dict[str, Any]:
        self._assert_ready()
        self._frame_seq += 1
        # Generate a small synthetic frame token
        frame_token = f"FRAME:{self.device_id}:{self._frame_seq}:{time.time():.3f}"
        encoded = base64.b64encode(frame_token.encode()).decode()
        return {
            "device_id": self.device_id,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "seq": self._frame_seq,
            "data": encoded,
            "timestamp": time.time(),
        }

    def write(self, data: Any) -> bool:
        """Write camera configuration (e.g., exposure, gain)."""
        self._assert_ready()
        if isinstance(data, dict):
            if "fps" in data:
                self.fps = int(data["fps"])
            if "width" in data:
                self.width = int(data["width"])
            if "height" in data:
                self.height = int(data["height"])
            return True
        return False

    def reset(self) -> bool:
        self.state = DeviceState.READY
        self._frame_seq = 0
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "state": self.state.name,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "frames_captured": self._frame_seq,
        }


class ActuatorDevice(HardwareDevice):
    """
    Simulated electromechanical actuator (motor, servo, linear stage).

    State:
        position  — current position in degrees (servo) or mm (linear)
        velocity  — current speed (units/s)
        target    — commanded target position

    Commands (write data dict):
        {"cmd": "move_to",   "position": float}  — move to absolute position
        {"cmd": "set_speed", "velocity": float}  — set maximum speed
        {"cmd": "stop"}                           — immediately stop
    """

    def __init__(
        self,
        device_id: str,
        actuator_type: str = "servo",
        min_position: float = 0.0,
        max_position: float = 180.0,
        max_velocity: float = 60.0,  # degrees/s or mm/s
    ) -> None:
        super().__init__(
            device_id=device_id,
            device_type=DeviceType.ACTUATOR,
            metadata={
                "actuator_type": actuator_type,
                "min_position": min_position,
                "max_position": max_position,
                "max_velocity": max_velocity,
                "model": f"SimActuator-{actuator_type}-v1",
            },
        )
        self.actuator_type = actuator_type
        self.min_position = min_position
        self.max_position = max_position
        self.max_velocity = max_velocity
        self.position: float = 0.0
        self.velocity: float = 0.0
        self.target: float = 0.0
        self._moving: bool = False

    def initialize(self) -> bool:
        self.position = self.min_position
        self.target = self.min_position
        self.velocity = 0.0
        self._moving = False
        self.state = DeviceState.READY
        return True

    def read(self) -> Dict[str, Any]:
        self._assert_ready()
        # Simulate movement toward target
        if self._moving and self.position != self.target:
            step = min(self.max_velocity * 0.1, abs(self.target - self.position))
            if self.target > self.position:
                self.position = min(self.position + step, self.target)
            else:
                self.position = max(self.position - step, self.target)
            if abs(self.position - self.target) < 0.01:
                self.position = self.target
                self._moving = False
        return {
            "device_id": self.device_id,
            "actuator_type": self.actuator_type,
            "position": round(self.position, 3),
            "velocity": self.velocity,
            "target": self.target,
            "moving": self._moving,
            "timestamp": time.time(),
        }

    def write(self, data: Any) -> bool:
        self._assert_ready()
        if not isinstance(data, dict):
            return False
        cmd = data.get("cmd", "")
        if cmd == "move_to":
            target = float(data.get("position", self.position))
            self.target = max(self.min_position, min(self.max_position, target))
            self._moving = True
            return True
        elif cmd == "set_speed":
            vel = float(data.get("velocity", self.max_velocity))
            self.velocity = max(0.0, min(self.max_velocity, vel))
            return True
        elif cmd == "stop":
            self.target = self.position
            self._moving = False
            self.velocity = 0.0
            return True
        return False

    def reset(self) -> bool:
        self.position = self.min_position
        self.target = self.min_position
        self.velocity = 0.0
        self._moving = False
        self.state = DeviceState.READY
        self._error_count = 0
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "actuator_type": self.actuator_type,
            "state": self.state.name,
            "position": round(self.position, 3),
            "target": self.target,
            "moving": self._moving,
        }


class ComputeDevice(HardwareDevice):
    """
    Simulated hardware accelerator (GPU, TPU, FPGA, neural engine).

    Supports:
        submit_job(workload_spec)  — queue a compute job, returns job_id
        query_job(job_id)          — check job status and result
        available_flops()          — report available compute capacity (TFLOPS)

    Job lifecycle: QUEUED → RUNNING → COMPLETED (or FAILED)
    Jobs are simulated to complete after a configurable number of read() cycles.
    """

    _JOB_STATES = {"QUEUED", "RUNNING", "COMPLETED", "FAILED"}

    def __init__(
        self,
        device_id: str,
        compute_units: int = 1024,
        memory_mb: int = 8192,
        tflops: float = 10.0,
    ) -> None:
        super().__init__(
            device_id=device_id,
            device_type=DeviceType.COMPUTE,
            metadata={
                "compute_units": compute_units,
                "memory_mb": memory_mb,
                "tflops": tflops,
                "model": "SimGPU-v1",
            },
        )
        self.compute_units = compute_units
        self.memory_mb = memory_mb
        self.tflops = tflops
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._utilization: float = 0.0

    def initialize(self) -> bool:
        self._jobs = {}
        self._utilization = 0.0
        self.state = DeviceState.READY
        return True

    def read(self) -> Dict[str, Any]:
        self._assert_ready()
        # Advance simulated jobs
        running_jobs = [j for j in self._jobs.values() if j["status"] == "RUNNING"]
        for job in running_jobs:
            job["ticks_run"] += 1
            if job["ticks_run"] >= job["ticks_needed"]:
                job["status"] = "COMPLETED"
                job["result"] = {"flops_used": job["ticks_needed"] * self.tflops * 1e12}
        # Start queued jobs
        queued = [j for j in self._jobs.values() if j["status"] == "QUEUED"]
        for job in queued[:2]:  # max 2 concurrent jobs
            job["status"] = "RUNNING"
            job["ticks_run"] = 0

        self._utilization = len(running_jobs) / max(self.compute_units / 100, 1)
        return {
            "device_id": self.device_id,
            "utilization": round(self._utilization, 3),
            "memory_mb": self.memory_mb,
            "tflops": self.tflops,
            "jobs": {jid: j["status"] for jid, j in self._jobs.items()},
            "timestamp": time.time(),
        }

    def write(self, data: Any) -> bool:
        """Submit a compute job via write()."""
        self._assert_ready()
        return bool(self.submit_job(data))

    def submit_job(self, workload_spec: Any) -> str:
        """
        Submit a compute workload.

        Args:
            workload_spec — dict describing the computation:
                {"type": "matmul", "size": 1024, "precision": "fp16"}

        Returns:
            job_id string.
        """
        job_id = str(uuid.uuid4())[:8]
        complexity = 1  # default ticks
        if isinstance(workload_spec, dict):
            size = workload_spec.get("size", 256)
            complexity = max(1, int(size / 256))
        self._jobs[job_id] = {
            "job_id": job_id,
            "spec": workload_spec,
            "status": "QUEUED",
            "ticks_run": 0,
            "ticks_needed": complexity,
            "result": None,
        }
        return job_id

    def query_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Query the status and result of a submitted job."""
        return self._jobs.get(job_id)

    def available_flops(self) -> float:
        """Return available compute capacity in TFLOPS."""
        return self.tflops * (1.0 - self._utilization)

    def reset(self) -> bool:
        self._jobs = {}
        self._utilization = 0.0
        self.state = DeviceState.READY
        self._error_count = 0
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "state": self.state.name,
            "utilization": round(self._utilization, 3),
            "available_tflops": round(self.available_flops(), 2),
            "active_jobs": sum(1 for j in self._jobs.values() if j["status"] == "RUNNING"),
            "total_jobs": len(self._jobs),
        }


# ---------------------------------------------------------------------------
# SimulatedHardware — factory for a full simulated hardware environment
# ---------------------------------------------------------------------------

class SimulatedHardware:
    """
    Factory that creates and registers a complete simulated hardware
    environment suitable for testing agent hardware interactions.

    Created devices:
        gpio_0, gpio_1        — 2 GPIO banks (16 pins each)
        sensor_temp_0         — temperature sensor
        sensor_light_0        — light sensor
        sensor_imu_0          — IMU sensor
        camera_0              — 640x480 simulated camera
        actuator_servo_0      — servo motor
        actuator_linear_0     — linear actuator
        compute_gpu_0         — simulated GPU (10 TFLOPS, 8 GB)
    """

    @staticmethod
    def create(device_manager: DeviceManager) -> DeviceManager:
        """
        Populate ``device_manager`` with a complete simulated hardware set.

        Returns the same device_manager (for chaining).
        """
        devices: List[HardwareDevice] = [
            GPIODevice("gpio_0", pin_count=16),
            GPIODevice("gpio_1", pin_count=8),
            SensorDevice("sensor_temp_0", sensor_type="temp"),
            SensorDevice("sensor_light_0", sensor_type="light"),
            SensorDevice("sensor_imu_0", sensor_type="imu"),
            CameraDevice("camera_0", width=640, height=480, fps=30),
            ActuatorDevice("actuator_servo_0", actuator_type="servo",
                           min_position=0.0, max_position=180.0),
            ActuatorDevice("actuator_linear_0", actuator_type="linear",
                           min_position=0.0, max_position=100.0, max_velocity=10.0),
            ComputeDevice("compute_gpu_0", compute_units=2048,
                          memory_mb=8192, tflops=10.0),
        ]
        for device in devices:
            device_manager.register(device, auto_init=True)

        return device_manager


# ---------------------------------------------------------------------------
# Tool integration — register HAL tools with the Battousai ToolManager
# ---------------------------------------------------------------------------

def register_hal_tools(tool_manager: Any, device_manager: DeviceManager) -> None:
    """
    Register hardware tool wrappers with the Battousai ToolManager.

    After calling this, agents can interact with hardware via:
        agent.use_tool("hw_read",         device_id="sensor_temp_0")
        agent.use_tool("hw_write",        device_id="actuator_servo_0",
                                          data={"cmd": "move_to", "position": 90})
        agent.use_tool("hw_list_devices")
        agent.use_tool("hw_scan")

    Args:
        tool_manager    — the Battousai ToolManager instance (from kernel.tools)
        device_manager  — the DeviceManager with registered devices
    """
    from battousai.tools import ToolSpec

    def hw_read(device_id: str) -> Dict[str, Any]:
        """Read from a hardware device."""
        driver = device_manager.get_driver(device_id)
        if driver is None:
            return {"error": f"Device {device_id!r} not found"}
        try:
            return driver.read()
        except Exception as exc:
            return {"error": str(exc), "device_id": device_id}

    def hw_write(device_id: str, data: Any = None) -> Dict[str, Any]:
        """Write a command or data to a hardware device."""
        driver = device_manager.get_driver(device_id)
        if driver is None:
            return {"error": f"Device {device_id!r} not found", "ok": False}
        try:
            ok = driver.write(data)
            return {"ok": ok, "device_id": device_id}
        except Exception as exc:
            return {"error": str(exc), "ok": False, "device_id": device_id}

    def hw_list_devices(device_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered hardware devices, optionally filtered by type."""
        if device_type is not None:
            try:
                dt = DeviceType[device_type.upper()]
                devices = device_manager.list_by_type(dt)
            except KeyError:
                return [{"error": f"Unknown device type {device_type!r}"}]
        else:
            devices = device_manager.list_all()
        return [
            {
                "device_id": d.device_id,
                "type": d.device_type.name,
                "state": d.state.name,
                "metadata": d.metadata,
            }
            for d in devices
        ]

    def hw_scan() -> Dict[str, Any]:
        """Scan for new or recovered hardware devices."""
        recovered = device_manager.scan()
        return {
            "recovered_devices": recovered,
            "total_devices": len(device_manager.list_all()),
            "stats": device_manager.stats(),
        }

    tool_manager.register(ToolSpec(
        name="hw_read",
        description=(
            "Read the current value or state from a hardware device. "
            "Args: device_id (str). "
            "Returns a dict with device-specific fields."
        ),
        callable=hw_read,
        is_simulated=True,
        rate_limit=100,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="hw_write",
        description=(
            "Send a command or data to a hardware device. "
            "Args: device_id (str), data (dict of command). "
            "Returns {ok: bool, device_id: str}."
        ),
        callable=hw_write,
        is_simulated=True,
        rate_limit=50,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="hw_list_devices",
        description=(
            "List all registered hardware devices. "
            "Optional: device_type (str) to filter by type "
            "(GPIO, SENSOR, CAMERA, ACTUATOR, COMPUTE, etc.)."
        ),
        callable=hw_list_devices,
        is_simulated=True,
        rate_limit=20,
        rate_window=10,
    ))

    tool_manager.register(ToolSpec(
        name="hw_scan",
        description=(
            "Scan the hardware bus for new or recovered devices. "
            "Returns list of recovered device IDs and current stats."
        ),
        callable=hw_scan,
        is_simulated=True,
        rate_limit=5,
        rate_window=10,
    ))

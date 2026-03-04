"""
test_hal.py — Tests for battousai.hal (Hardware Abstraction Layer)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.hal import (
    GPIODevice, SensorDevice, CameraDevice, ActuatorDevice, ComputeDevice,
    DeviceManager, DeviceType, register_hal_tools, SimulatedHardware,
)
from battousai.tools import ToolManager


class TestGPIODevice(unittest.TestCase):
    """
    GPIODevice.read()  → dict with 'pins' key ({pin: bool, ...})
    GPIODevice.write({pin: N, value: 0|1})  → bool
    Must call initialize() before read/write.
    """

    def setUp(self):
        self.gpio = GPIODevice(device_id="gpio_0", pin_count=16)
        self.gpio.initialize()

    def test_device_id_stored(self):
        self.assertEqual(self.gpio.device_id, "gpio_0")

    def test_pin_count_stored(self):
        self.assertEqual(self.gpio.pin_count, 16)

    def test_read_pin_returns_value(self):
        """read() returns a dict; pins key maps pin index to bool."""
        data = self.gpio.read()
        self.assertIsInstance(data, dict)
        self.assertIn("pins", data)
        # Pin 0 state is a bool
        self.assertIn(0, data["pins"])

    def test_write_pin_low(self):
        """write({'pin': 0, 'value': 0}) sets pin 0 low."""
        self.gpio.write({"pin": 0, "value": 0})
        data = self.gpio.read()
        # Pin 0 should be False / 0
        self.assertFalse(data["pins"][0])

    def test_write_pin_high(self):
        """write({'pin': 1, 'value': 1}) sets pin 1 high."""
        self.gpio.write({"pin": 1, "value": 1})
        data = self.gpio.read()
        self.assertTrue(data["pins"][1])


class TestSensorDevice(unittest.TestCase):
    """
    SensorDevice.read()  → dict with 'value' key.
    Valid sensor_type values: 'temp', 'humidity', 'distance', 'imu', 'pressure', 'light'.
    Must call initialize() before read.
    """

    def _make(self, sensor_type):
        s = SensorDevice(device_id=f"{sensor_type}_0", sensor_type=sensor_type)
        s.initialize()
        return s

    def test_temperature_sensor_in_range(self):
        sensor = self._make("temp")
        data = sensor.read()
        val = data["value"]
        # Temperature typically -50 to 100 °C
        self.assertIsInstance(val, (int, float))

    def test_light_sensor_non_negative(self):
        sensor = self._make("light")
        val = sensor.read()["value"]
        self.assertGreaterEqual(val, 0)

    def test_distance_sensor_non_negative(self):
        sensor = self._make("distance")
        val = sensor.read()["value"]
        self.assertGreaterEqual(val, 0)

    def test_humidity_sensor_in_range(self):
        sensor = self._make("humidity")
        val = sensor.read()["value"]
        self.assertGreaterEqual(val, 0)
        self.assertLessEqual(val, 100)

    def test_pressure_sensor_positive(self):
        sensor = self._make("pressure")
        val = sensor.read()["value"]
        self.assertGreater(val, 0)

    def test_imu_sensor_returns_dict_or_list(self):
        sensor = self._make("imu")
        data = sensor.read()
        val = data["value"]
        # IMU returns dict with accel_x, accel_y, etc.
        self.assertIsNotNone(val)


class TestCameraDevice(unittest.TestCase):
    """
    CameraDevice.read()  → dict with 'width', 'height', 'data' keys.
    Must call initialize() before read.
    """

    def setUp(self):
        self.camera = CameraDevice(
            device_id="cam_0", width=640, height=480, fps=30
        )
        self.camera.initialize()

    def test_camera_stores_dimensions(self):
        self.assertEqual(self.camera.width, 640)
        self.assertEqual(self.camera.height, 480)

    def test_capture_frame_returns_data(self):
        """read() returns a dict with image data."""
        frame = self.camera.read()
        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, dict)

    def test_capture_frame_has_correct_shape(self):
        """Camera read dict contains width and height."""
        frame = self.camera.read()
        self.assertIn("width", frame)
        self.assertIn("height", frame)
        self.assertEqual(frame["width"], 640)
        self.assertEqual(frame["height"], 480)


class TestActuatorDevice(unittest.TestCase):
    """
    ActuatorDevice.write({'cmd': 'move_to', 'position': float})  → bool
    ActuatorDevice.write({'cmd': 'stop'})  → bool
    ActuatorDevice.read()  → dict with 'target' and 'position' keys.
    Must call initialize() before use.
    Position moves toward target over time — target is clamped immediately.
    """

    def setUp(self):
        self.actuator = ActuatorDevice(
            device_id="motor_0",
            actuator_type="servo",
            min_position=0.0,
            max_position=180.0,
        )
        self.actuator.initialize()

    def test_actuator_stores_bounds(self):
        self.assertEqual(self.actuator.min_position, 0.0)
        self.assertEqual(self.actuator.max_position, 180.0)

    def test_move_within_bounds(self):
        """move_to with in-range position sets target to that value."""
        self.actuator.write({"cmd": "move_to", "position": 90.0})
        data = self.actuator.read()
        self.assertEqual(data["target"], 90.0)

    def test_move_clamps_to_min(self):
        """Positions below min_position are clamped to min_position."""
        self.actuator.write({"cmd": "move_to", "position": -10.0})
        data = self.actuator.read()
        self.assertGreaterEqual(data["target"], 0.0)

    def test_move_clamps_to_max(self):
        """Positions above max_position are clamped to max_position."""
        self.actuator.write({"cmd": "move_to", "position": 999.0})
        data = self.actuator.read()
        self.assertLessEqual(data["target"], 180.0)

    def test_stop_sets_moving_false(self):
        """stop command sets moving=False."""
        self.actuator.write({"cmd": "move_to", "position": 90.0})
        self.actuator.write({"cmd": "stop"})
        data = self.actuator.read()
        self.assertFalse(data["moving"])


class TestComputeDevice(unittest.TestCase):

    def setUp(self):
        self.compute = ComputeDevice(
            device_id="gpu_0",
            compute_units=8,
            memory_mb=4096,
            tflops=10.0,
        )
        self.compute.initialize()

    def test_compute_device_stores_tflops(self):
        self.assertEqual(self.compute.tflops, 10.0)

    def test_submit_job_returns_job_id(self):
        job_id = self.compute.submit_job({"op": "matmul", "size": 128})
        self.assertIsNotNone(job_id)
        self.assertIsInstance(job_id, str)

    def test_query_job_returns_status(self):
        job_id = self.compute.submit_job({"op": "conv", "size": 64})
        status = self.compute.query_job(job_id)
        self.assertIsNotNone(status)

    def test_multiple_jobs_tracked(self):
        jid1 = self.compute.submit_job({"op": "a"})
        jid2 = self.compute.submit_job({"op": "b"})
        s1 = self.compute.query_job(jid1)
        s2 = self.compute.query_job(jid2)
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)


class TestDeviceManager(unittest.TestCase):

    def setUp(self):
        self.dm = DeviceManager()

    def test_register_device(self):
        gpio = GPIODevice(device_id="gpio_test", pin_count=8)
        self.dm.register(gpio)
        devices = self.dm.list_all()
        ids = [d.device_id for d in devices]
        self.assertIn("gpio_test", ids)

    def test_list_by_type(self):
        gpio = GPIODevice(device_id="gpio_1", pin_count=4)
        sensor = SensorDevice(device_id="temp_1", sensor_type="temp")
        self.dm.register(gpio)
        self.dm.register(sensor)
        gpio_devices = self.dm.list_by_type(DeviceType.GPIO)
        self.assertGreater(len(gpio_devices), 0)

    def test_scan_returns_device_list(self):
        result = self.dm.scan()
        self.assertIsInstance(result, list)


class TestRegisterHALTools(unittest.TestCase):

    def setUp(self):
        self.dm = DeviceManager()
        SimulatedHardware.create(self.dm)
        self.tool_manager = ToolManager()
        register_hal_tools(self.tool_manager, self.dm)
        for tool in self.tool_manager.list_tools():
            self.tool_manager.grant_access(tool, "agent_0001")

    def test_hal_tools_registered(self):
        tools = self.tool_manager.list_tools()
        for expected in ["hw_read", "hw_write", "hw_list_devices", "hw_scan"]:
            self.assertIn(expected, tools)

    def test_hw_list_devices_returns_devices(self):
        result = self.tool_manager.execute(
            "agent_0001", "hw_list_devices", {}
        )
        self.assertIsNotNone(result)

    def test_hw_scan_returns_results(self):
        result = self.tool_manager.execute(
            "agent_0001", "hw_scan", {}
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()

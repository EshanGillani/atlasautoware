"""
Hardware-layer unit tests — no ROS, no hardware, just the pure logic.
=====================================================================

Covers the parts of the driver stack that can silently wreck a run if the
maths is wrong: PCA9685 register sequencing + pulse conversion, Ackermann ->
pulse mapping, the VESC packet framing/CRC/telemetry offsets, drive-backend
auto-detection, RPLidar scan binning, and the IMU traction governor.

    python3 -m pytest tests/test_hardware.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'f1tenth_gym_ros'))
import pca9685 as pca                                    # noqa: E402
import vesc_protocol as vp                               # noqa: E402
from rplidar_node import bin_scan                        # noqa: E402
from drive_node import pick_backend                      # noqa: E402
from mpc_controller import TractionGovernor              # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# PCA9685
# ─────────────────────────────────────────────────────────────────────────────

class FakeBus:
    """Records register writes; answers reads with the last written value."""

    def __init__(self):
        self.regs = {}
        self.writes = []

    def write_byte_data(self, addr, reg, val):
        self.regs[reg] = val
        self.writes.append((reg, val))

    def read_byte_data(self, addr, reg):
        return self.regs.get(reg, 0)

    def write_i2c_block_data(self, addr, reg, data):
        self.writes.append((reg, list(data)))
        for i, b in enumerate(data):
            self.regs[reg + i] = b


def test_pca9685_prescale_50hz():
    bus = FakeBus()
    pca.PCA9685(bus, 0x40, 50.0)
    # 25 MHz / (4096 * 50) - 1 = 121
    assert bus.regs[pca.PCA9685.PRESCALE] == 121


def test_pulse_to_ticks():
    assert pca.us_to_ticks(1500, 50.0) == 307            # mid pulse at 50 Hz
    assert pca.us_to_ticks(0, 50.0) == 0


def test_set_pulse_writes_channel_registers():
    bus = FakeBus()
    dev = pca.PCA9685(bus, 0x40, 50.0)
    dev.set_pulse_us(2, 1500)
    reg, data = bus.writes[-1]
    assert reg == pca.PCA9685.LED0_ON_L + 8              # channel 2
    assert data == [0, 0, 307 & 0xFF, 307 >> 8]


def test_speed_to_us_mapping():
    assert pca.speed_to_us(0.0, 7.0) == 1500.0
    assert pca.speed_to_us(7.0, 7.0) == 2000.0
    assert pca.speed_to_us(-7.0, 7.0) == 1000.0
    assert pca.speed_to_us(3.5, 7.0) == 1750.0
    assert pca.speed_to_us(99.0, 7.0) == 2000.0          # clipped


def test_steer_to_us_mapping():
    assert pca.steer_to_us(0.0, 0.41) == 1500.0
    assert pca.steer_to_us(0.41, 0.41) == 1900.0
    assert pca.steer_to_us(-0.41, 0.41) == 1100.0
    assert pca.steer_to_us(0.41, 0.41, invert=True) == 1100.0
    assert pca.steer_to_us(0.0, 0.41, trim_us=25.0) == 1525.0


# ─────────────────────────────────────────────────────────────────────────────
# VESC protocol
# ─────────────────────────────────────────────────────────────────────────────

def test_crc16_xmodem_vector():
    assert vp.crc16(b'123456789') == 0x31C3              # standard check value


def test_frame_roundtrip():
    payload = bytes([vp.COMM_SET_RPM, 0, 0, 0x12, 0x34])
    pkt = vp.frame(payload)
    assert pkt[0] == 0x02 and pkt[-1] == 0x03
    parser = vp.PacketParser()
    # split across feeds + leading garbage: parser must resync and reassemble
    out = parser.feed(b'\xff\x00' + pkt[:4])
    out += parser.feed(pkt[4:] + vp.frame(b'\x04'))
    assert out == [payload, b'\x04']


def test_parser_rejects_corrupt_crc():
    pkt = bytearray(vp.frame(b'\x04\x01\x02'))
    pkt[3] ^= 0xFF                                       # flip a payload byte
    assert vp.PacketParser().feed(bytes(pkt)) == []


def test_parse_values_layout():
    import struct
    payload = bytearray(60)
    payload[0] = vp.COMM_GET_VALUES
    struct.pack_into('>h', payload, 1, 312)              # temp_fet 31.2 C
    struct.pack_into('>h', payload, 21, -500)            # duty -0.5
    struct.pack_into('>i', payload, 23, 9228)            # erpm
    struct.pack_into('>h', payload, 27, 118)             # 11.8 V
    struct.pack_into('>i', payload, 45, 12345)           # tachometer
    payload[53] = 0
    v = vp.parse_values(bytes(payload))
    assert v['temp_fet'] == 31.2
    assert v['duty'] == -0.5
    assert v['erpm'] == 9228.0
    assert v['v_in'] == 11.8
    assert v['tachometer'] == 12345
    assert vp.parse_values(b'\x05') is None              # wrong command id


def test_command_packets_decode():
    parser = vp.PacketParser()
    (p,) = parser.feed(vp.pkt_set_rpm(-9228))
    assert p[0] == vp.COMM_SET_RPM
    import struct
    assert struct.unpack('>i', p[1:])[0] == -9228
    (p,) = parser.feed(vp.pkt_set_servo_pos(0.75))
    assert p[0] == vp.COMM_SET_SERVO_POS
    assert struct.unpack('>H', p[1:])[0] == 750


# ─────────────────────────────────────────────────────────────────────────────
# Drive backend auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def _cfg():
    return {'max_speed': 7.0, 'max_steer': 0.41, 'steer_invert': False,
            'steer_trim_us': 0.0, 'i2c_bus': 1, 'i2c_address': 0x40,
            'pwm_hz': 50.0, 'throttle_channel': 0, 'steer_channel': 1,
            'neutral_us': 1500.0, 'full_fwd_us': 2000.0, 'full_rev_us': 1000.0,
            'steer_center_us': 1500.0, 'steer_half_range_us': 400.0,
            'serial_port': '/dev/null-nonexistent', 'serial_baud': 115200,
            'erpm_gain': 4614.0}


def test_pick_backend_none_when_no_hardware():
    # neither a PCA9685 on i2c nor a VESC on serial exists in CI
    assert pick_backend('auto', _cfg(), lambda m: None) is None
    assert pick_backend('pca9685', _cfg(), lambda m: None) is None
    assert pick_backend('vesc', _cfg(), lambda m: None) is None


def test_pca_backend_commands(monkeypatch):
    import drive_node
    bus = FakeBus()
    monkeypatch.setattr(drive_node.pca, 'open_i2c', lambda n: bus)
    backend = pick_backend('pca9685', _cfg(), lambda m: None)
    assert backend is not None and backend.name == 'pca9685'
    backend.command(3.5, 0.41)                           # half throttle, full left
    writes = [w for w in bus.writes if isinstance(w[1], list)]
    thr_reg = pca.PCA9685.LED0_ON_L
    str_reg = pca.PCA9685.LED0_ON_L + 4
    thr = [d for r, d in writes if r == thr_reg][-1]
    st = [d for r, d in writes if r == str_reg][-1]
    assert thr[2] | (thr[3] << 8) == pca.us_to_ticks(1750, 50.0)
    assert st[2] | (st[3] << 8) == pca.us_to_ticks(1900, 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# RPLidar binning
# ─────────────────────────────────────────────────────────────────────────────

def test_bin_scan_geometry():
    # RPLidar 0 deg (front, clockwise) -> ROS angle 0 -> middle bin of [-pi, pi)
    meas = [(15, 0.0, 2000.0),       # 2 m dead ahead
            (15, 90.0, 1000.0),      # 90 deg CW = ROS -pi/2 (right)
            (15, 270.0, 3000.0),     # 270 CW = ROS +pi/2 (left)
            (0, 180.0, 4000.0),      # quality 0 -> dropped
            (15, 45.0, 50.0)]        # 5 cm -> below range_min, dropped
    r = bin_scan(meas, num_bins=360, range_min=0.1, range_max=25.0)
    inc = 2.0 * math.pi / 360
    idx = lambda th: int((th + math.pi) / inc) % 360     # noqa: E731
    assert r[idx(0.0)] == np.float32(2.0)
    assert r[idx(-math.pi / 2)] == np.float32(1.0)
    assert r[idx(math.pi / 2)] == np.float32(3.0)
    assert np.isinf(r[idx(math.pi - 0.01)])              # nothing behind


def test_bin_scan_nearest_wins():
    meas = [(15, 0.0, 5000.0), (15, 0.1, 2000.0)]        # same bin, keep 2 m
    r = bin_scan(meas, num_bins=180)
    assert r.min() == np.float32(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Traction governor
# ─────────────────────────────────────────────────────────────────────────────

def test_governor_unity_within_grip():
    gov = TractionGovernor(max_lat_accel=6.0)
    for _ in range(50):
        scale = gov.update(yaw_rate=1.0, speed=3.0)      # a_lat = 3 < 6
    assert scale == 1.0


def test_governor_cuts_speed_past_grip_and_recovers():
    gov = TractionGovernor(max_lat_accel=6.0, alpha=1.0, min_scale=0.6)
    scale = gov.update(yaw_rate=3.0, speed=4.0)          # a_lat = 12 = 2x limit
    assert abs(scale - 0.6) < 1e-9                       # floored at min_scale
    for _ in range(100):
        scale = gov.update(yaw_rate=0.5, speed=2.0)      # back under the limit
    assert scale == 1.0


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))

"""
VESC UART protocol — framing, commands, telemetry parsing (pure, no ROS).
=========================================================================

Minimal implementation of the VESC (bldc firmware) serial protocol, enough to
drive the car and read telemetry without the C++ vesc driver stack:

  packet     = 0x02 | len(1B) | payload | crc16-xmodem(payload) | 0x03
  commands   COMM_SET_RPM / SET_DUTY / SET_CURRENT / SET_SERVO_POS
  telemetry  COMM_GET_VALUES -> erpm, duty, input voltage, currents, tacho

The GET_VALUES field layout matches the unified FW 3.x-6.x firmware (the one
the official VESC Tool ships); if your readings look wrong, update the
firmware rather than the offsets.  Everything here is pure bytes-in/bytes-out
and unit-tested in tests/test_hardware.py; the serial port handling lives in
drive_node.py.
"""

import struct

COMM_FW_VERSION = 0
COMM_GET_VALUES = 4
COMM_SET_DUTY = 5
COMM_SET_CURRENT = 6
COMM_SET_CURRENT_BRAKE = 7
COMM_SET_RPM = 8
COMM_SET_SERVO_POS = 12


def crc16(data):
    """CRC-16/XMODEM (poly 0x1021, init 0) as used by the VESC firmware."""
    crc = 0
    for b in bytes(data):
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
        crc &= 0xFFFF
    return crc


def frame(payload):
    """Wrap a command payload in a short VESC packet (payload < 256 bytes)."""
    payload = bytes(payload)
    return (b'\x02' + struct.pack('>B', len(payload)) + payload
            + struct.pack('>H', crc16(payload)) + b'\x03')


def pkt_set_rpm(erpm):
    return frame(struct.pack('>Bi', COMM_SET_RPM, int(erpm)))


def pkt_set_duty(duty):
    return frame(struct.pack('>Bi', COMM_SET_DUTY, int(round(duty * 1e5))))


def pkt_set_current(amps):
    return frame(struct.pack('>Bi', COMM_SET_CURRENT, int(round(amps * 1e3))))


def pkt_set_current_brake(amps):
    return frame(struct.pack('>Bi', COMM_SET_CURRENT_BRAKE,
                             int(round(amps * 1e3))))


def pkt_set_servo_pos(pos):
    """Steering servo on the VESC's own servo header, pos in [0, 1]."""
    pos = max(0.0, min(1.0, float(pos)))
    return frame(struct.pack('>BH', COMM_SET_SERVO_POS, int(round(pos * 1e3))))


def pkt_request(comm_id):
    return frame(struct.pack('>B', comm_id))


class PacketParser:
    """Incremental decoder: feed raw serial bytes, collect verified payloads."""

    def __init__(self):
        self._buf = b''

    def feed(self, data):
        self._buf += bytes(data)
        payloads = []
        while True:
            start = self._buf.find(b'\x02')
            if start < 0:
                self._buf = b''
                break
            self._buf = self._buf[start:]
            if len(self._buf) < 5:
                break
            n = self._buf[1]
            end = 2 + n + 3                      # len byte + payload + crc + 0x03
            if len(self._buf) < end:
                break
            payload = self._buf[2:2 + n]
            crc = struct.unpack('>H', self._buf[2 + n:4 + n])[0]
            if self._buf[end - 1] == 0x03 and crc == crc16(payload):
                payloads.append(payload)
                self._buf = self._buf[end:]
            else:                                # corrupt — resync past this 0x02
                self._buf = self._buf[1:]
        return payloads


def parse_values(payload):
    """COMM_GET_VALUES payload -> dict (unified FW 3.x-6.x layout)."""
    if not payload or payload[0] != COMM_GET_VALUES:
        return None
    u = struct.unpack_from
    return {
        'temp_fet': u('>h', payload, 1)[0] / 1e1,
        'temp_motor': u('>h', payload, 3)[0] / 1e1,
        'current_motor': u('>i', payload, 5)[0] / 1e2,
        'current_input': u('>i', payload, 9)[0] / 1e2,
        'duty': u('>h', payload, 21)[0] / 1e3,
        'erpm': float(u('>i', payload, 23)[0]),
        'v_in': u('>h', payload, 27)[0] / 1e1,
        'tachometer': u('>i', payload, 45)[0],
        'fault': payload[53] if len(payload) > 53 else 0,
    }

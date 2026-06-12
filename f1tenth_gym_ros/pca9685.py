"""
PCA9685 16-channel PWM driver + Ackermann pulse mapping (pure, no ROS).
=======================================================================

Register-level driver for the PCA9685 I2C PWM board used as the actuation
path on the car: throttle pulses into the VESC's **PPM input** and (optionally)
servo pulses for steering.  Configure the VESC's PPM app and run its pulse
calibration so 1000/1500/2000 us map to full-brake/neutral/full-throttle.

Takes any smbus-compatible bus object (write_byte_data / read_byte_data /
write_i2c_block_data), so the register sequencing and the unit->pulse maths
are unit-testable with a fake bus — see tests/test_hardware.py.  The ROS node
that drives this lives in drive_node.py.
"""

import time


class PCA9685:
    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    OSC_HZ = 25_000_000.0

    def __init__(self, bus, address=0x40, freq_hz=50.0):
        self.bus = bus
        self.addr = address
        self.freq_hz = float(freq_hz)
        bus.write_byte_data(self.addr, self.MODE2, 0x04)        # totem-pole output
        bus.write_byte_data(self.addr, self.MODE1, 0x20)        # auto-increment
        self.set_pwm_freq(self.freq_hz)

    def set_pwm_freq(self, hz):
        self.freq_hz = float(hz)
        prescale = int(round(self.OSC_HZ / (4096.0 * self.freq_hz))) - 1
        prescale = max(3, min(255, prescale))
        old = self.bus.read_byte_data(self.addr, self.MODE1)
        self.bus.write_byte_data(self.addr, self.MODE1, (old & 0x7F) | 0x10)  # sleep
        self.bus.write_byte_data(self.addr, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.addr, self.MODE1, old)
        time.sleep(0.0005)
        self.bus.write_byte_data(self.addr, self.MODE1, old | 0xA0)          # restart

    def set_pulse_us(self, channel, us):
        ticks = us_to_ticks(us, self.freq_hz)
        reg = self.LED0_ON_L + 4 * int(channel)
        self.bus.write_i2c_block_data(
            self.addr, reg, [0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F])

    def set_off(self, channel):
        reg = self.LED0_ON_L + 4 * int(channel)
        self.bus.write_i2c_block_data(self.addr, reg, [0, 0, 0, 0x10])  # full OFF


def us_to_ticks(us, freq_hz):
    """Pulse width in microseconds -> PCA9685 12-bit off-tick count."""
    return int(round(float(us) * float(freq_hz) * 4096.0 / 1e6))


def speed_to_us(speed, max_speed, neutral_us=1500.0,
                full_fwd_us=2000.0, full_rev_us=1000.0):
    """m/s -> throttle pulse, linear about neutral, clipped to full scale."""
    frac = max(-1.0, min(1.0, float(speed) / float(max_speed)))
    span = (full_fwd_us - neutral_us) if frac >= 0.0 else (neutral_us - full_rev_us)
    return neutral_us + frac * span


def steer_to_us(angle, max_steer, center_us=1500.0, half_range_us=400.0,
                invert=False, trim_us=0.0):
    """rad -> servo pulse.  `trim_us` corrects mechanical centre offset."""
    frac = max(-1.0, min(1.0, float(angle) / float(max_steer)))
    if invert:
        frac = -frac
    return center_us + trim_us + frac * half_range_us


def open_i2c(bus_num):
    """Open /dev/i2c-N via smbus2 (preferred) or smbus."""
    try:
        from smbus2 import SMBus
    except ImportError:
        from smbus import SMBus  # noqa: F401 — system python3-smbus fallback
    return SMBus(int(bus_num))

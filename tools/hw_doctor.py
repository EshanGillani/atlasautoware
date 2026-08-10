"""
hw_doctor — check every connection before you drive.
====================================================

Debugging a dead car on the grid costs you the session.  This walks the whole
chain the car depends on — Python packages, the I2C bus and the PCA9685, the
VESC over UART, the RPLidar, the OAK-D, the ROS graph, and the map/raceline
files — and for anything broken prints the specific command that fixes it.

    python3 tools/hw_doctor.py              # everything, human readable
    python3 tools/hw_doctor.py --fix        # add the exact fix for each failure
    python3 tools/hw_doctor.py --json       # for the dashboard
    python3 tools/hw_doctor.py --only vesc,lidar
    python3 tools/hw_doctor.py --config config/hardware.yaml

Every probe is read-only and passive: it opens ports, reads registers and
listens.  Nothing here ever commands throttle, so it is safe to run with the
car powered, on the ground, wheels down.

Exit code is 0 if nothing FAILED (warnings are fine), 1 otherwise — so it can
gate a launch script.
"""

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'f1tenth_gym_ros'))

OK, WARN, FAIL, SKIP = 'ok', 'warn', 'fail', 'skip'

C = dict(bold='\033[1m', dim='\033[2m', red='\033[31m', grn='\033[32m',
         ylw='\033[33m', cyn='\033[36m', off='\033[0m')
if os.name == 'nt' and not os.environ.get('WT_SESSION'):
    C = {k: '' for k in C}
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

MARK = {OK: ('grn', 'PASS'), WARN: ('ylw', 'WARN'),
        FAIL: ('red', 'FAIL'), SKIP: ('dim', 'skip')}


class Report(list):
    """Collected results; each is (group, name, status, detail, fix)."""

    def add(self, group, name, status, detail='', fix=''):
        self.append(dict(group=group, name=name, status=status,
                         detail=detail, fix=fix))
        return status

    def failed(self):
        return [r for r in self if r['status'] == FAIL]


# ── config ───────────────────────────────────────────────────────────────────
def load_config(path):
    """hardware.yaml -> {node: {param: value}}; {} if unreadable."""
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return {node: (body or {}).get('ros__parameters', {})
            for node, body in raw.items() if isinstance(body, dict)}


def param(cfg, node, key, default):
    return cfg.get(node, {}).get(key, default)


def off_car(port):
    """True when this machine cannot possibly hold the car's hardware.

    The doctor is meant to run on the Jetson, but you also run it on a Windows
    laptop to check racelines and configs before a session.  There, a POSIX
    device path is not a failure — there is simply no car attached — so those
    probes skip rather than shouting FAIL at someone who is nowhere near it.
    """
    return sys.platform.startswith('win') and str(port).startswith('/dev/')


# ── probes ───────────────────────────────────────────────────────────────────
def check_python(rep, cfg):
    g = 'python'
    v = sys.version_info
    rep.add(g, 'interpreter', OK if v >= (3, 8) else WARN,
            f'{v.major}.{v.minor}.{v.micro} at {sys.executable}',
            'the stack targets Python 3.8+')

    # (module, why it matters, is it fatal on the car, how to get it)
    mods = [
        ('numpy',   'all control maths',            True,  'pip3 install numpy'),
        ('yaml',    'reading config + map files',   True,  'pip3 install pyyaml'),
        ('scipy',   'raceline optimization',        False, 'pip3 install scipy'),
        ('osqp',    'the MPC solver (falls back to MAP without it)',
         False, 'pip3 install osqp==0.6.3'),
        ('serial',  'VESC UART backend',            False, 'pip3 install pyserial'),
        ('smbus2',  'PCA9685 I2C backend',          False, 'pip3 install smbus2'),
        ('rplidar', 'the lidar driver',             False,
         'pip3 install rplidar-roboticia'),
        ('depthai', 'the OAK-D camera',             False, 'pip3 install depthai'),
        ('cv2',     'camera perception',            False,
         'pip3 install opencv-python'),
        ('torch',   'the RL policy',                False, 'pip3 install torch'),
        ('rclpy',   'everything ROS',               False,
         'source /opt/ros/humble/setup.bash'),
    ]
    for mod, why, fatal, fix in mods:
        try:
            m = __import__(mod)
            ver = getattr(m, '__version__', '')
            rep.add(g, mod, OK, f'{ver}  {why}'.strip())
        except Exception:
            rep.add(g, mod, FAIL if fatal else WARN, f'missing — {why}', fix)


def check_i2c(rep, cfg):
    """I2C bus reachable, and does the PCA9685 answer at its address?"""
    g = 'pca9685'
    bus_num = int(param(cfg, 'drive_node', 'i2c_bus', 1))
    addr = int(param(cfg, 'drive_node', 'i2c_address', 0x40))

    if sys.platform.startswith('win'):
        return rep.add(g, 'i2c bus', SKIP, 'no I2C on Windows — check on the Jetson')

    node = f'/dev/i2c-{bus_num}'
    if not os.path.exists(node):
        buses = sorted(glob.glob('/dev/i2c-*'))
        return rep.add(g, 'i2c bus', FAIL,
                       f'{node} does not exist'
                       + (f' (found {", ".join(buses)})' if buses else ''),
                       f'check `i2cdetect -l` and set drive_node.i2c_bus '
                       f'in config/hardware.yaml')
    if not os.access(node, os.R_OK | os.W_OK):
        return rep.add(g, 'i2c bus', FAIL, f'{node} exists but is not writable',
                       f'sudo usermod -aG i2c $USER   (then log out and back in)')
    rep.add(g, 'i2c bus', OK, node)

    try:
        from smbus2 import SMBus
    except ImportError:
        return rep.add(g, 'pca9685', SKIP, 'smbus2 not installed',
                       'pip3 install smbus2')
    try:
        with SMBus(bus_num) as bus:
            mode1 = bus.read_byte_data(addr, 0x00)      # MODE1 — a passive read
        rep.add(g, 'pca9685', OK,
                f'responds at 0x{addr:02x} (MODE1=0x{mode1:02x})')
    except Exception as e:
        rep.add(g, 'pca9685', WARN,
                f'nothing at 0x{addr:02x} on bus {bus_num}: {e}',
                'run `i2cdetect -y {0}` — if the board shows at another address, '
                'set drive_node.i2c_address. Harmless if you use the VESC-UART '
                'backend instead.'.format(bus_num))


def check_vesc(rep, cfg):
    """Open the VESC serial port and ask for telemetry — a real handshake."""
    g = 'vesc'
    port = param(cfg, 'drive_node', 'serial_port', '/dev/ttyACM0')
    baud = int(param(cfg, 'drive_node', 'serial_baud', 115200))

    if off_car(port):
        return rep.add(g, 'port', SKIP,
                       f'{port} is a Linux device path — check this on the car')

    try:
        import serial
    except ImportError:
        return rep.add(g, 'port', SKIP, 'pyserial not installed',
                       'pip3 install pyserial')

    if not sys.platform.startswith('win') and not os.path.exists(port):
        acms = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
        return rep.add(g, 'port', WARN,
                       f'{port} not present'
                       + (f' (serial devices present: {", ".join(acms)})' if acms
                          else ' (no serial devices at all)'),
                       'check the USB cable and `dmesg | tail`; set '
                       'drive_node.serial_port to whichever device appeared. '
                       'Harmless if you actuate through the PCA9685 instead.')

    try:
        import vesc_protocol as vp
    except Exception as e:
        return rep.add(g, 'protocol', FAIL, f'cannot import vesc_protocol: {e}')

    try:
        with serial.Serial(port, baud, timeout=0.5) as ser:
            ser.reset_input_buffer()
            ser.write(vp.pkt_request(vp.COMM_GET_VALUES))
            ser.flush()
            parser = vp.PacketParser()
            values, deadline = None, time.time() + 1.0
            while time.time() < deadline and values is None:
                chunk = ser.read(256)
                if chunk:
                    for payload in parser.feed(chunk):
                        values = vp.parse_values(payload) or values
                else:
                    time.sleep(0.02)
    except Exception as e:
        return rep.add(g, 'port', FAIL, f'could not open {port}: {e}',
                       'sudo usermod -aG dialout $USER  (then log out and back '
                       'in), and make sure VESC Tool is not holding the port')

    rep.add(g, 'port', OK, f'{port} open at {baud} baud')
    if values is None:
        return rep.add(g, 'telemetry', WARN,
                       'port opened but the VESC did not answer GET_VALUES',
                       'in VESC Tool set App to UART at the same baud, and '
                       'confirm the firmware is 3.x-6.x')

    rep.add(g, 'telemetry', OK,
            f"v_in={values['v_in']:.1f}V  erpm={values['erpm']:.0f}  "
            f"fet={values['temp_fet']:.0f}C  fault={values['fault']}")
    # Battery state is the single most common cause of a car that "drives badly".
    v = values['v_in']
    if v < 1.0:
        rep.add(g, 'battery', WARN, f'{v:.1f} V — no pack connected?',
                'plug the battery in; the VESC reads ~0 V on USB power alone')
    elif v < 11.1:                                   # 3S nominal floor
        rep.add(g, 'battery', WARN, f'{v:.1f} V — low',
                'charge the pack; grip and top speed fall off as it sags')
    else:
        rep.add(g, 'battery', OK, f'{v:.1f} V')
    if values['fault']:
        rep.add(g, 'fault', FAIL, f"VESC fault code {values['fault']}",
                'read the fault in VESC Tool -> Terminal `faults` before driving')


def check_lidar(rep, cfg):
    g = 'lidar'
    port = param(cfg, 'rplidar_node', 'port', '/dev/ttyUSB0')
    baud = int(param(cfg, 'rplidar_node', 'baudrate', 115200))

    if off_car(port):
        return rep.add(g, 'port', SKIP,
                       f'{port} is a Linux device path — check this on the car')

    if not sys.platform.startswith('win') and not os.path.exists(port):
        usbs = sorted(glob.glob('/dev/ttyUSB*'))
        return rep.add(g, 'port', FAIL,
                       f'{port} not present'
                       + (f' (found {", ".join(usbs)})' if usbs else ''),
                       'plug in the lidar; set rplidar_node.port to the device '
                       'that appears in `dmesg | tail`')
    try:
        from rplidar import RPLidar
    except ImportError:
        return rep.add(g, 'driver', SKIP, 'rplidar package not installed',
                       'pip3 install rplidar-roboticia')

    lidar = None
    try:
        lidar = RPLidar(port, baudrate=baud, timeout=2.0)
        info = lidar.get_info()
        health = lidar.get_health()
        rep.add(g, 'device', OK,
                f"model {info.get('model')} fw {info.get('firmware')} "
                f"hw {info.get('hardware')}")
        status = str(health[0]).lower() if health else 'unknown'
        if status.startswith('good'):
            rep.add(g, 'health', OK, 'Good')
        else:
            rep.add(g, 'health', FAIL, f'health reports {health}',
                    'power-cycle the lidar; if it stays bad the motor or the '
                    'optics need attention')
    except Exception as e:
        rep.add(g, 'device', FAIL, f'no response on {port} at {baud}: {e}',
                f'A1/A2 use 115200, A3/S1 use 256000 — set '
                f'rplidar_node.baudrate to match your unit, and check the '
                f'5V supply (a browning-out lidar answers nothing)')
    finally:
        if lidar is not None:
            try:
                lidar.stop()
                lidar.disconnect()
            except Exception:
                pass


def check_camera(rep, cfg):
    g = 'camera'                      # must match the key in CHECKS for the header
    try:
        import depthai as dai
    except ImportError:
        return rep.add(g, 'depthai', SKIP, 'depthai not installed',
                       'pip3 install depthai')
    try:
        devices = dai.Device.getAllAvailableDevices()
    except Exception as e:
        return rep.add(g, 'device', FAIL, f'depthai enumeration failed: {e}',
                       'replug the camera; OAK-D needs USB3 for full frame rate')
    if not devices:
        return rep.add(g, 'device', WARN, 'no OAK-D found',
                       'check the USB3 cable (a USB2 cable enumerates but '
                       'throttles); on Linux install the udev rules: '
                       'echo \'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", '
                       'MODE="0666"\' | sudo tee /etc/udev/rules.d/80-movidius.rules'
                       ' && sudo udevadm control --reload-rules')
    for d in devices:
        rep.add(g, 'device', OK,
                f'{getattr(d, "getMxId", lambda: "?")()} '
                f'({getattr(d, "state", "")})')


def check_ros(rep, cfg):
    """Is a ROS graph up, and are the topics the racing node needs alive?"""
    g = 'ros'
    try:
        import rclpy                                    # noqa: F401
    except Exception:
        return rep.add(g, 'rclpy', SKIP, 'ROS 2 not sourced here',
                       'source /opt/ros/humble/setup.bash')

    import rclpy
    from rclpy.node import Node
    wanted = {
        param(cfg, 'raceline_mpc', 'scan_topic', '/scan'): 'lidar scans',
        param(cfg, 'raceline_mpc', 'odom_topic', '/pf/pose/odom'): 'localization pose',
        param(cfg, 'raceline_mpc', 'drive_topic', '/drive'): 'drive commands',
        param(cfg, 'raceline_mpc', 'imu_topic', '/oakd/imu'): 'IMU (traction governor)',
    }
    started = False
    try:
        if not rclpy.ok():
            rclpy.init()
            started = True
        node = Node('hw_doctor')
        deadline = time.time() + 2.0                    # let discovery settle
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        live = {name: types for name, types in node.get_topic_names_and_types()}
        for topic, why in wanted.items():
            if not topic:
                continue
            if topic in live:
                n = node.count_publishers(topic)
                rep.add(g, topic, OK if n else WARN,
                        f'{why} — {n} publisher(s)' if n
                        else f'{why} — advertised but nobody is publishing')
            else:
                rep.add(g, topic, WARN, f'{why} — not on the graph',
                        'start the bringup: atlas run car')
        node.destroy_node()
    except Exception as e:
        rep.add(g, 'graph', WARN, f'could not inspect the ROS graph: {e}')
    finally:
        if started:
            try:
                rclpy.shutdown()
            except Exception:
                pass


def check_files(rep, cfg):
    """The data the car cannot start without: maps, racelines, configs."""
    g = 'files'
    for name in ('config/hardware.yaml', 'config/sim.yaml', 'config/slam_mapping.yaml'):
        p = os.path.join(REPO, name)
        if not os.path.exists(p):
            rep.add(g, name, FAIL, 'missing')
            continue
        try:
            import yaml
            with open(p) as f:
                yaml.safe_load(f)
            rep.add(g, name, OK, 'parses')
        except ImportError:
            rep.add(g, name, OK, 'present (pyyaml missing, not parsed)')
        except Exception as e:
            rep.add(g, name, FAIL, f'will not parse: {e}',
                    'fix the YAML — the launch files read this at startup')

    maps = glob.glob(os.path.join(REPO, 'maps', '*.yaml'))
    rep.add(g, 'maps', OK if maps else FAIL,
            f'{len(maps)} map(s): ' + ', '.join(sorted(
                os.path.basename(m) for m in maps)) if maps else 'no maps found',
            '' if maps else 'run a mapping session: atlas run map-session')

    lines = sorted(glob.glob(os.path.join(REPO, 'racelines', '*.csv')))
    if not lines:
        return rep.add(g, 'racelines', FAIL, 'no raceline CSVs',
                       'generate one: atlas run optimize')
    for p in lines:
        ok, detail = validate_raceline(p)
        rep.add(g, 'raceline ' + os.path.basename(p), OK if ok else FAIL, detail,
                '' if ok else 'regenerate it: atlas run optimize')


def validate_raceline(path):
    """A raceline is only usable if it is finite, closed and sanely profiled."""
    try:
        import csv
        xs, ys, sp = [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                xs.append(float(row['x']))
                ys.append(float(row['y']))
                sp.append(float(row['speed']))
    except Exception as e:
        return False, f'unreadable: {e}'
    if len(xs) < 50:
        return False, f'only {len(xs)} points — too short to drive'
    if any(v != v for v in xs + ys + sp):
        return False, 'contains NaN'
    if min(sp) <= 0.0:
        return False, f'speed floor {min(sp):.2f} m/s — the car would stop'
    if max(sp) > 15.0:
        return False, f'speed peak {max(sp):.1f} m/s — implausible for 1/10 scale'
    gap = ((xs[0] - xs[-1]) ** 2 + (ys[0] - ys[-1]) ** 2) ** 0.5
    closed = gap < 2.0
    return True, (f'{len(xs)} pts, v {min(sp):.1f}-{max(sp):.1f} m/s, '
                  f'{"closed" if closed else f"OPEN loop (gap {gap:.1f} m)"}')


CHECKS = [
    ('python',  'Python packages',      check_python),
    ('pca9685', 'I2C / PCA9685',        check_i2c),
    ('vesc',    'VESC (motor + ESC)',   check_vesc),
    ('lidar',   'RPLidar',              check_lidar),
    ('camera',  'OAK-D camera',         check_camera),
    ('ros',     'ROS 2 graph',          check_ros),
    ('files',   'Maps and racelines',   check_files),
]


def run(only=None, config=None):
    cfg = load_config(config or os.path.join(REPO, 'config', 'hardware.yaml'))
    rep = Report()
    for key, _label, fn in CHECKS:
        if only and key not in only:
            continue
        try:
            fn(rep, cfg)
        except Exception as e:                       # a probe must never abort the run
            rep.add(key, 'probe', FAIL, f'check itself crashed: {e}')
    return rep


def render(rep, show_fix):
    labels = dict((k, l) for k, l, _ in CHECKS)
    print()
    seen = []
    for r in rep:
        if r['group'] not in seen:
            seen.append(r['group'])
            print(f"{C['bold']}{labels.get(r['group'], r['group'])}{C['off']}")
        col, word = MARK[r['status']]
        print(f"  {C[col]}{word}{C['off']}  {r['name']:<28} {r['detail']}")
        if r['fix'] and (show_fix or r['status'] == FAIL):
            for line in _wrap(r['fix'], 64):
                print(f"        {C['dim']}-> {line}{C['off']}")
        if r is rep[-1] or rep[rep.index(r) + 1]['group'] != r['group']:
            print()

    counts = {s: sum(1 for r in rep if r['status'] == s)
              for s in (OK, WARN, FAIL, SKIP)}
    bad = counts[FAIL]
    print(f"{C['bold']}Summary{C['off']}  "
          f"{C['grn']}{counts[OK]} pass{C['off']}  "
          f"{C['ylw']}{counts[WARN]} warn{C['off']}  "
          f"{C['red']}{counts[FAIL]} fail{C['off']}  "
          f"{C['dim']}{counts[SKIP]} skipped{C['off']}")
    if bad:
        print(f"\n{C['red']}Not ready to drive.{C['off']} Fix the failures above"
              + ('' if show_fix else ' (re-run with --fix for the commands).'))
    else:
        print(f"\n{C['grn']}Clear to drive.{C['off']} Start slow: "
              f"atlas run race -- -p v_scale:=0.3")
    print()


def _wrap(text, width):
    words, line, out = str(text).split(), '', []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f'{line} {w}'.strip()
    if line:
        out.append(line)
    return out or ['']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', default='',
                    help='comma-separated subset: '
                         + ','.join(k for k, _, _ in CHECKS))
    ap.add_argument('--config', default='', help='hardware YAML to read')
    ap.add_argument('--fix', action='store_true',
                    help='show the fix hint for warnings too, not just failures')
    ap.add_argument('--json', action='store_true', help='machine-readable output')
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(',') if s.strip()} or None
    if only:
        unknown = only - {k for k, _, _ in CHECKS}
        if unknown:
            raise SystemExit(f'unknown check(s): {", ".join(sorted(unknown))}')

    rep = run(only, args.config or None)
    if args.json:
        print(json.dumps({'checks': list(rep),
                          'ok': not rep.failed(),
                          'ts': time.time()}, indent=2))
    else:
        render(rep, args.fix)
    return 1 if rep.failed() else 0


if __name__ == '__main__':
    sys.exit(main())

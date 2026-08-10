"""
Launcher, registry and hardware doctor.
=======================================

The registry is the index the CLI and the dashboard both read, so a broken
entry does not fail loudly — it produces a button that silently runs the wrong
thing, or nothing. These tests check every entry actually points at a file that
exists and is described well enough to be usable under time pressure.

The doctor tests cover the logic that decides *pass / warn / fail*, since that
verdict is what someone reads before deciding it is safe to power the motor.

No ROS, no hardware.

    python3 -m pytest tests/test_atlas.py -q
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))

import atlas                                              # noqa: E402
import atlas_registry as reg                              # noqa: E402
import hw_doctor as doc                                   # noqa: E402


# ── registry integrity ───────────────────────────────────────────────────────
class TestRegistry:
    def test_ids_are_unique(self):
        ids = [c['id'] for c in reg.COMMANDS]
        assert len(ids) == len(set(ids)), \
            f'duplicates: {[i for i in ids if ids.count(i) > 1]}'

    def test_every_command_has_the_required_fields(self):
        for c in reg.COMMANDS:
            for field in ('id', 'group', 'kind', 'target', 'env', 'where',
                          'title'):
                assert c.get(field), f"{c.get('id')} is missing {field}"

    def test_field_values_are_from_the_known_sets(self):
        groups = {g[0] for g in reg.GROUPS}
        for c in reg.COMMANDS:
            assert c['group'] in groups, f"{c['id']}: bad group {c['group']}"
            assert c['kind'] in ('launch', 'node', 'python', 'shell')
            assert c['env'] in ('ros', 'gym', 'plain')
            assert c['where'] in ('sim', 'car', 'either', 'host')

    def test_python_and_shell_targets_exist_on_disk(self):
        """A registry entry pointing at a missing file is a dead button."""
        missing = []
        for c in reg.COMMANDS:
            if c['kind'] not in ('python', 'shell'):
                continue
            if c['target'].startswith('-m '):             # module, not a path
                continue
            if not os.path.exists(os.path.join(REPO, *c['target'].split('/'))):
                missing.append(f"{c['id']} -> {c['target']}")
        assert not missing, 'missing targets: ' + ', '.join(missing)

    def test_ros_nodes_are_registered_as_entry_points(self):
        """`ros2 run` only finds executables declared in setup.py."""
        with open(os.path.join(REPO, 'setup.py')) as f:
            setup = f.read()
        for c in reg.COMMANDS:
            if c['kind'] == 'node':
                assert f"{c['target']} =" in setup, \
                    f"{c['id']}: '{c['target']}' is not an entry point in setup.py"

    def test_launch_files_exist(self):
        for c in reg.COMMANDS:
            if c['kind'] == 'launch':
                p = os.path.join(REPO, 'launch', c['target'])
                assert os.path.exists(p), f"{c['id']}: no {p}"

    def test_car_moving_commands_are_flagged_dangerous(self):
        """The flag drives the CLI confirmation and the dashboard's guard."""
        for cid in ('car', 'race', 'race-strategy', 'rl-drive', 'map-drive'):
            assert reg.get(cid).get('danger'), f'{cid} should be flagged danger'

    def test_every_command_explains_itself(self):
        """These are read by someone under pressure at a competition."""
        for c in reg.COMMANDS:
            assert len(c['title']) > 12, f"{c['id']}: title too terse"
            assert len(c.get('long', '')) > 60, f"{c['id']}: needs a real explanation"

    def test_argument_specs_are_well_formed(self):
        for c in reg.COMMANDS:
            for entry in c.get('args', []):
                assert len(entry) == 3, f"{c['id']}: args must be (flag, help, default)"
                flag, help_, _default = entry
                assert flag and help_, f"{c['id']}: empty flag or help"

    def test_advertised_flags_exist_in_the_target_script(self):
        """The registry populates the CLI help and the dashboard's form fields,
        so a flag that the script does not accept is a form field that produces
        an error, and a stale default is one that silently does the wrong thing.

        Checked by reading each script's own `add_argument` calls, so the two
        cannot drift apart unnoticed.
        """
        import re
        problems = []
        for c in reg.COMMANDS:
            if c['kind'] != 'python' or c['target'].startswith('-m '):
                continue
            path = os.path.join(REPO, *c['target'].split('/'))
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                source = f.read()
            declared = set(re.findall(r"add_argument\(\s*'(--[\w-]+)'", source))
            if not declared:                      # script takes no options
                continue
            for flag, _help, _default in c.get('args', []):
                name = flag.split()[0].rstrip(':=').rstrip('=')
                if not name.startswith('--'):     # positional or ROS -p param
                    continue
                if name not in declared:
                    problems.append(f"{c['id']}: {name} not accepted by "
                                    f"{c['target']}")
        assert not problems, '\n'.join(problems)

    def test_advertised_choices_are_valid(self):
        """A default that is not among the script's choices fails at launch."""
        import re
        problems = []
        for c in reg.COMMANDS:
            if c['kind'] != 'python' or c['target'].startswith('-m '):
                continue
            path = os.path.join(REPO, *c['target'].split('/'))
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                source = f.read()
            for flag, _help, default in c.get('args', []):
                if not default or not flag.startswith('--'):
                    continue
                m = re.search(
                    r"add_argument\(\s*'" + re.escape(flag) +
                    r"'[^)]*choices=\[([^\]]*)\]", source)
                if not m:
                    continue
                choices = {s.strip().strip('\'"') for s in m.group(1).split(',')}
                if default not in choices:
                    problems.append(f"{c['id']}: default '{default}' for {flag} "
                                    f"is not in {sorted(choices)}")
        assert not problems, '\n'.join(problems)

    def test_lookup_helpers_agree_with_the_table(self):
        assert set(reg.ids()) == {c['id'] for c in reg.COMMANDS}
        assert reg.get('doctor')['id'] == 'doctor'
        assert reg.get('nope') is None
        grouped = sum(len(cmds) for _g, _l, _b, cmds in reg.by_group())
        assert grouped == len(reg.COMMANDS), 'a command fell out of every group'


# ── environment resolution ───────────────────────────────────────────────────
class TestContextChoice:
    BARE = dict(native_ros=False, container=None, docker=False, gym=False,
                numpy=True, torch=False, workspace=None)

    def test_plain_commands_run_locally(self):
        cmd = dict(env='plain', needs=[])
        ctx, _why = atlas.choose_context(cmd, dict(self.BARE))
        assert ctx == 'local'

    def test_ros_commands_prefer_native_over_docker(self):
        env = dict(self.BARE, native_ros=True, container='sim-1')
        ctx, _ = atlas.choose_context(dict(env='ros', needs=[]), env)
        assert ctx == 'native'

    def test_ros_commands_fall_back_to_the_container(self):
        env = dict(self.BARE, container='f1tenth_gym_ros-sim-1')
        ctx, _ = atlas.choose_context(dict(env='ros', needs=[]), env)
        assert ctx == 'docker'

    def test_unavailable_context_explains_why(self):
        ctx, why = atlas.choose_context(dict(env='ros', needs=[]), dict(self.BARE))
        assert ctx is None
        assert 'ROS' in why and len(why) > 20

    def test_gym_commands_need_gym_or_a_container(self):
        assert atlas.choose_context(dict(env='gym', needs=[]),
                                    dict(self.BARE))[0] is None
        assert atlas.choose_context(dict(env='gym', needs=[]),
                                    dict(self.BARE, gym=True))[0] == 'local'

    def test_missing_packages_are_named(self):
        cmd = dict(env='plain', needs=['definitely_not_a_real_module'])
        ctx, why = atlas.choose_context(cmd, dict(self.BARE))
        assert ctx is None
        assert 'definitely_not_a_real_module' in why


# ── command construction ─────────────────────────────────────────────────────
class TestBuild:
    ENV = dict(native_ros=False, container='sim-1', docker=True, gym=False,
               numpy=True, torch=False, workspace=None)

    def test_ros_node_becomes_ros2_run(self):
        cmd = reg.get('race')
        argv, shown = atlas.build(cmd, [], dict(self.ENV), 'docker')
        assert 'ros2 run f1tenth_gym_ros raceline_mpc' in shown
        assert argv[:3] == ['docker', 'exec', '-it']

    def test_launch_file_becomes_ros2_launch(self):
        argv, shown = atlas.build(reg.get('sim'), [], dict(self.ENV), 'docker')
        assert 'ros2 launch f1tenth_gym_ros gym_bridge_launch.py' in shown

    def test_local_python_uses_this_interpreter(self):
        """So a virtualenv is respected instead of whatever python3 resolves to."""
        argv, _ = atlas.build(reg.get('draw'), [], dict(self.ENV), 'local')
        assert argv[0] == sys.executable

    def test_extra_arguments_are_passed_through_verbatim(self):
        argv, shown = atlas.build(reg.get('optimize'), ['--margin', '0.3'],
                                  dict(self.ENV), 'local')
        assert argv[-2:] == ['--margin', '0.3']
        assert '--margin 0.3' in shown

    def test_module_targets_run_as_dash_m(self):
        argv, _ = atlas.build(reg.get('tests'), [], dict(self.ENV), 'local')
        assert argv[1] == '-m' and argv[2] == 'pytest'

    def test_paths_are_native_for_this_platform(self):
        argv, _ = atlas.build(reg.get('draw'), [], dict(self.ENV), 'local')
        assert os.sep in argv[1]
        if os.sep != '/':
            assert '/' not in argv[1], 'mixed separators leaked into the path'


# ── hardware doctor ──────────────────────────────────────────────────────────
class TestDoctor:
    def test_report_tracks_failures(self):
        rep = doc.Report()
        rep.add('g', 'fine', doc.OK, 'all good')
        rep.add('g', 'meh', doc.WARN, 'optional thing missing')
        assert not rep.failed()
        rep.add('g', 'bad', doc.FAIL, 'broken', 'do this')
        assert len(rep.failed()) == 1

    def test_config_files_are_read(self):
        cfg = doc.load_config(os.path.join(REPO, 'config', 'hardware.yaml'))
        assert 'drive_node' in cfg
        assert doc.param(cfg, 'drive_node', 'serial_port', 'x') == '/dev/ttyACM0'

    def test_missing_config_degrades_quietly(self):
        assert doc.load_config('/no/such/file.yaml') == {}
        assert doc.param({}, 'node', 'key', 'fallback') == 'fallback'

    def test_off_car_only_skips_posix_devices_on_windows(self):
        is_win = sys.platform.startswith('win')
        assert doc.off_car('/dev/ttyACM0') is is_win
        assert doc.off_car('COM3') is False

    def test_a_real_raceline_validates(self):
        ok, detail = doc.validate_raceline(
            os.path.join(REPO, 'racelines', 'comp_raceline.csv'))
        assert ok, detail
        assert 'closed' in detail

    def test_bad_racelines_are_rejected(self, tmp_path):
        """These are the ways a generated line kills a car."""
        cases = {
            'nan': 'x,y,heading,curvature,speed\n' +
                   ''.join(f'{i},0,0,0,nan\n' for i in range(60)),
            'too_short': 'x,y,heading,curvature,speed\n1,1,0,0,3\n',
            'zero_speed': 'x,y,heading,curvature,speed\n' +
                          ''.join(f'{i},0,0,0,0\n' for i in range(60)),
            'absurd_speed': 'x,y,heading,curvature,speed\n' +
                            ''.join(f'{i},0,0,0,99\n' for i in range(60)),
        }
        for name, body in cases.items():
            p = tmp_path / f'{name}.csv'
            p.write_text(body)
            ok, _detail = doc.validate_raceline(str(p))
            assert not ok, f'{name} should have been rejected'

    def test_unreadable_raceline_is_reported_not_raised(self):
        ok, detail = doc.validate_raceline('/no/such/raceline.csv')
        assert not ok and 'unreadable' in detail

    def test_python_check_never_fails_on_optional_packages(self):
        """Missing depthai must not read as 'the car is broken'."""
        rep = doc.Report()
        doc.check_python(rep, {})
        for r in rep:
            if r['name'] in ('depthai', 'rplidar', 'cv2', 'torch', 'smbus2'):
                assert r['status'] != doc.FAIL, f"{r['name']} must not be fatal"

    def test_file_check_finds_maps_and_racelines(self):
        rep = doc.Report()
        doc.check_files(rep, {})
        names = [r['name'] for r in rep]
        assert 'maps' in names
        assert any(n.startswith('raceline ') for n in names)

    def test_every_check_is_wired_up(self):
        for key, label, fn in doc.CHECKS:
            assert callable(fn) and label
            rep = doc.Report()
            fn(rep, doc.load_config(
                os.path.join(REPO, 'config', 'hardware.yaml')))
            assert all(r['group'] == key for r in rep), \
                f"{key}: results must be tagged with the check's own key"

    def test_a_crashing_probe_does_not_abort_the_run(self):
        """One bad probe must not stop you seeing the other twenty."""
        original = doc.CHECKS[:]
        try:
            def boom(rep, cfg):
                raise RuntimeError('sensor exploded')
            doc.CHECKS.append(('files', 'Boom', boom))
            rep = doc.run(only={'files'})
            assert any('crashed' in r['detail'] for r in rep)
        finally:
            doc.CHECKS[:] = original


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))

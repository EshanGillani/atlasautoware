"""
atlas — one command to run anything in this stack.
==================================================

The problem this solves: the stack has ROS nodes, ROS launch files, gym-only
scripts and plain-Python tools, and each needs a different incantation depending
on whether you are on the Jetson, on a Linux workstation with ROS, or on a
Windows laptop driving a Docker container.  Remembering which is which under
time pressure at a competition is how sessions get wasted.

So: describe every script once (tools/atlas_registry.py), detect the environment
once (here), and let one verb run all of it.

    python3 tools/atlas.py list                 # everything you can run
    python3 tools/atlas.py info race            # what it does, which flags
    python3 tools/atlas.py run sim              # start the simulator
    python3 tools/atlas.py run race -- -p v_scale:=0.3
    python3 tools/atlas.py doctor               # check the hardware
    python3 tools/atlas.py env                  # what did it detect?

Everything after `--` is passed through untouched.

Environment resolution
----------------------
Each command declares what it needs (`ros`, `gym`, or `plain`).  atlas picks the
first context that can satisfy it:

    native   ROS sourced on this machine (the Jetson, a Linux workstation)
    docker   the sim container, with this repo bind-mounted
    local    plain Python on this machine (works fine on Windows)

`atlas env` prints exactly what was found, which is the first thing to check
when something will not start.
"""

import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_registry as reg

# Where the repo is mounted inside the sim container (docker-compose.yml).
CONTAINER_REPO = '/sim_ws/src/f1tenth_gym_ros'
ROS_SETUP = '/opt/ros/humble/setup.bash'
# Workspace overlays to try sourcing, in order; the first that exists wins.
WS_CANDIDATES = [
    '/sim_ws/install/setup.bash',
    os.path.expanduser('~/sim_ws/install/setup.bash'),
    os.path.expanduser('~/f1tenth_ws/install/setup.bash'),
    os.path.join(REPO, '..', '..', 'install', 'setup.bash'),
]

C = dict(bold='\033[1m', dim='\033[2m', red='\033[31m', grn='\033[32m',
         ylw='\033[33m', cyn='\033[36m', off='\033[0m')
if os.name == 'nt' and not os.environ.get('WT_SESSION'):
    C = {k: '' for k in C}          # legacy conhost: no escapes

# The registry text uses en/em dashes; a cp1252 console would raise on them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:               # not a real tty, or too old to reconfigure
        pass


def c(key, s):
    return f"{C[key]}{s}{C['off']}"


# ── environment detection ────────────────────────────────────────────────────
def has_native_ros():
    return os.path.exists(ROS_SETUP)


def workspace_setup():
    for p in WS_CANDIDATES:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def sim_container():
    """Name of a running sim container, or None."""
    if not shutil.which('docker'):
        return None
    try:
        p = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'],
                           capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    names = [n.strip() for n in p.stdout.splitlines() if n.strip()]
    for n in names:
        if 'f1tenth' in n or 'sim' in n:
            return n
    return None


def can_import(mod):
    """Is a module importable in THIS interpreter (used for `plain`/`gym`)?"""
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def detect():
    """One snapshot of the runtime environment, reused by every code path."""
    ws = workspace_setup()
    return {
        'native_ros': has_native_ros(),
        'workspace': ws,
        'container': sim_container(),
        'docker': bool(shutil.which('docker')),
        'gym': can_import('f110_gym') or can_import('gym'),
        'numpy': can_import('numpy'),
        'torch': can_import('torch'),
        'python': sys.executable,
        'platform': sys.platform,
        'repo': REPO,
    }


def choose_context(cmd, envinfo):
    """Where should this command run?  -> ('native'|'docker'|'local', reason)."""
    need = cmd['env']
    if need == 'ros':
        if envinfo['native_ros']:
            return 'native', 'ROS 2 Humble is sourced on this machine'
        if envinfo['container']:
            return 'docker', f"running inside container {envinfo['container']}"
        return None, ('needs ROS 2 — no /opt/ros/humble here and no sim '
                      'container running (start one with `docker-compose up`)')
    if need == 'gym':
        if envinfo['gym']:
            return 'local', 'f110_gym is importable here'
        if envinfo['container']:
            return 'docker', f"f110_gym lives in {envinfo['container']}"
        # Be specific: f110_gym pins gym==0.19.0 and numpy<=1.22.0, neither of
        # which has wheels for a recent Python, so "just pip install it" is
        # actively misleading advice on 3.12+. The container exists precisely
        # because it carries a Python the legacy pins still resolve against.
        hint = ''
        if sys.version_info >= (3, 12):
            hint = (f' (it pins gym==0.19.0 and numpy<=1.22.0, which have no '
                    f'wheels for your Python '
                    f'{sys.version_info.major}.{sys.version_info.minor} — '
                    f'use the container)')
        return None, ('needs f110_gym: start the sim container with '
                      '`docker-compose up -d`' + hint)
    # plain
    missing = [m for m in cmd.get('needs', []) if not can_import(m)]
    if not missing:
        return 'local', 'plain Python, runs here'
    if envinfo['container']:
        return 'docker', f"missing {', '.join(missing)} here"
    return None, f"missing Python packages: {', '.join(missing)} (pip install them)"


# ── command construction ─────────────────────────────────────────────────────
def inner_command(cmd, extra, repo, posix=False):
    """The command line as it runs *inside* the chosen context.

    `posix` forces forward-slash paths regardless of the host OS, which the
    docker context always needs: the command is assembled on this machine but
    executed inside a Linux container.
    """
    kind, target = cmd['kind'], cmd['target']
    if kind == 'launch':
        parts = ['ros2', 'launch', 'f1tenth_gym_ros', target]
    elif kind == 'node':
        parts = ['ros2', 'run', 'f1tenth_gym_ros', target]
    elif kind == 'shell':
        parts = ['bash', _join(repo, target, posix)]
    else:                                            # python
        if target.startswith('-m '):                 # e.g. "-m pytest"
            parts = ['python3', '-m', target.split(None, 1)[1]]
        else:
            parts = ['python3', _join(repo, target, posix)]
    return parts + list(extra)


def _join(repo, target, posix=False):
    """Join a repo root and a forward-slash registry target.

    Native separators for anything running on this machine; POSIX when the
    result is destined for the Linux container.  Normalising a container path
    with os.path.normpath on a Windows host produced
    `\\sim_ws\\src\\...\\train_duel.py`, which the container cannot open — so
    the two cases genuinely cannot share one code path.
    """
    if posix or os.sep == '/':
        return repo.rstrip('/') + '/' + target.lstrip('/')
    return os.path.normpath(os.path.join(repo, *target.split('/')))


def ros_prefix(ws):
    src = f'source {ROS_SETUP}; '
    if ws:
        src += f'source {ws} 2>/dev/null; '
    return src


def build(cmd, extra, envinfo, ctx):
    """-> (argv_for_subprocess, human_readable_string)."""
    if ctx == 'native':
        ws = envinfo['workspace']
        inner = ' '.join(inner_command(cmd, extra, REPO))
        line = ros_prefix(ws) + f'cd {REPO}; ' + inner
        return ['bash', '-lc', line], inner

    if ctx == 'docker':
        inner = ' '.join(inner_command(cmd, extra, CONTAINER_REPO, posix=True))
        line = (f'source {ROS_SETUP}; source /sim_ws/install/setup.bash 2>/dev/null; '
                f'cd {CONTAINER_REPO}; ' + inner)
        return (['docker', 'exec', '-it', envinfo['container'], 'bash', '-lc', line],
                inner)

    # local — no shell, no ROS; use THIS interpreter so venvs are respected
    parts = inner_command(cmd, extra, REPO)
    if parts[0] == 'python3':
        parts[0] = sys.executable
    elif parts[0] == 'bash' and os.name == 'nt' and not shutil.which('bash'):
        raise SystemExit(c('red', 'this command is a shell script and there is '
                                  'no bash on PATH — run it in the container'))
    return parts, ' '.join(parts[1:]) if parts else ''


# ── verbs ────────────────────────────────────────────────────────────────────
def cmd_env(args):
    e = detect()
    print(c('bold', '\nEnvironment\n'))
    rows = [
        ('platform',        e['platform']),
        ('python',          e['python']),
        ('repo',            e['repo']),
        ('ROS 2 Humble',    c('grn', 'yes') if e['native_ros'] else c('dim', 'no')),
        ('workspace overlay', e['workspace'] or c('dim', 'none found')),
        ('docker',          c('grn', 'yes') if e['docker'] else c('dim', 'no')),
        ('sim container',   c('grn', e['container']) if e['container']
                            else c('dim', 'not running')),
        ('f110_gym',        c('grn', 'yes') if e['gym'] else c('dim', 'no')),
        ('numpy',           c('grn', 'yes') if e['numpy'] else c('red', 'no')),
        ('torch (for RL)',  c('grn', 'yes') if e['torch'] else c('ylw', 'no')),
    ]
    for k, v in rows:
        print(f'  {k:<20} {v}')

    print(c('bold', '\nWhat this means\n'))
    if e['native_ros']:
        print('  ROS commands run natively here.')
    elif e['container']:
        print(f"  ROS commands run inside {e['container']} via docker exec.")
    else:
        print(c('ylw', '  ROS commands cannot run: no local ROS and no container.'))
        print('  Start one with:  docker-compose up -d')
    if not e['torch']:
        print(c('dim', '  RL training/deployment needs torch: pip install torch'))
    print()
    return 0


def cmd_list(args):
    e = detect()
    print(c('bold', '\nAtlas — everything you can run\n'))
    for gid, label, blurb, cmds in reg.by_group():
        if args.group and args.group != gid:
            continue
        if not cmds:
            continue
        print(f"{c('bold', label)}  {c('dim', blurb)}")
        for cmd in cmds:
            ctx, _ = choose_context(cmd, e)
            if ctx is None:
                mark = c('red', ' x')
            elif cmd.get('danger'):
                mark = c('ylw', ' !')
            else:
                mark = c('grn', ' o')
            print(f"  {mark} {c('cyn', cmd['id']):<28} {cmd['title']}")
        print()
    print(c('dim', '  o ready    ! moves a real car    x environment missing '
                   '(see `atlas env`)'))
    print(c('dim', '  details:  atlas info <id>        run:  atlas run <id> '
                   '[-- extra args]\n'))
    return 0


def cmd_info(args):
    cmd = reg.get(args.id)
    if not cmd:
        return unknown(args.id)
    e = detect()
    ctx, why = choose_context(cmd, e)
    print()
    print(f"{c('bold', cmd['id'])}  —  {cmd['title']}")
    print(c('dim', '  ' + '-' * 68))
    for line in _wrap(cmd.get('long', ''), 68):
        print('  ' + line)
    print()
    print(f"  {'runs on':<12} {cmd['where']}")
    print(f"  {'needs':<12} {cmd['env']}"
          + (f"  +  {', '.join(cmd['needs'])}" if cmd.get('needs') else ''))
    if ctx:
        print(f"  {'context':<12} {c('grn', ctx)}  {c('dim', '(' + why + ')')}")
    else:
        print(f"  {'context':<12} {c('red', 'unavailable')}  {c('dim', why)}")
    if cmd.get('danger'):
        print(f"  {'safety':<12} {c('ylw', 'this can move a real car')}")
    if cmd.get('args'):
        print(f"\n  {c('bold', 'arguments')}")
        for flag, help_, default in cmd['args']:
            d = f"  [{default}]" if default else ''
            print(f"    {c('cyn', flag):<28} {help_}{c('dim', d)}")
    print(f"\n  {c('bold', 'run it')}\n    python3 tools/atlas.py run {cmd['id']}\n")
    return 0


def cmd_run(args):
    cmd = reg.get(args.id)
    if not cmd:
        return unknown(args.id)
    e = detect()
    ctx, why = choose_context(cmd, e)
    if ctx is None:
        print(c('red', f"\ncannot run '{cmd['id']}': {why}\n"))
        print(c('dim', '  `atlas env` shows what was detected.\n'))
        return 2

    argv, shown = build(cmd, args.extra, e, ctx)

    if cmd.get('danger') and not args.yes and not args.dry_run:
        print(c('ylw', f"\n'{cmd['id']}' can move a real car."))
        print(f"  {shown}")
        try:
            if input('  type y to continue: ').strip().lower() != 'y':
                print('  cancelled.\n')
                return 1
        except (EOFError, KeyboardInterrupt):
            print('\n  cancelled.\n')
            return 1

    # flush: the child inherits our stdout, so an unflushed banner would land
    # after the child's output whenever atlas is piped.
    print(f"\n{c('dim', '[' + ctx + ']')} {c('cyn', shown)}\n", flush=True)
    if args.dry_run:
        print(c('dim', '  (dry run — not executed)\n'))
        print('  ' + ' '.join(argv) + '\n')
        return 0
    try:
        return subprocess.call(argv, cwd=REPO if ctx == 'local' else None)
    except KeyboardInterrupt:
        print(c('dim', '\n  interrupted\n'))
        return 130
    except FileNotFoundError as exc:
        print(c('red', f'  could not launch: {exc}\n'))
        return 2


def cmd_doctor(args):
    """Delegate to the hardware doctor, passing anything extra straight on."""
    doctor = os.path.join(REPO, 'tools', 'hw_doctor.py')
    if not os.path.exists(doctor):
        print(c('red', 'tools/hw_doctor.py is missing'))
        return 2
    return subprocess.call([sys.executable, doctor] + list(args.extra), cwd=REPO)


def unknown(name):
    print(c('red', f"\nunknown command '{name}'"))
    close = [i for i in reg.ids() if name in i or i in name]
    if close:
        print('  did you mean: ' + ', '.join(c('cyn', i) for i in close))
    print(c('dim', '  `atlas list` shows everything.\n'))
    return 2


def _wrap(text, width):
    """Small greedy wrapper — keeps the CLI dependency-free."""
    words, line, out = text.split(), '', []
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
    ap = argparse.ArgumentParser(
        prog='atlas', description='Run any part of the AtlasAutoware stack.')
    sub = ap.add_subparsers(dest='verb')

    p = sub.add_parser('list', help='show everything you can run')
    p.add_argument('group', nargs='?', help='only this group (sim, car, ...)')
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser('info', help='what a command does')
    p.add_argument('id')
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser('run', help='run a command')
    p.add_argument('id')
    p.add_argument('-n', '--dry-run', action='store_true',
                   help='print the command instead of running it')
    p.add_argument('-y', '--yes', action='store_true',
                   help='skip the confirmation on car-moving commands')
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('doctor', help='check hardware and connections')
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser('env', help='what environment was detected')
    p.set_defaults(fn=cmd_env)

    # Split on the first '--' ourselves.  argparse.REMAINDER would swallow
    # atlas's own flags (`run foo -n -- ...` lost the -n), and parse_known_args
    # alone would reorder pass-through flags.  Everything after '--' is the
    # target script's, untouched; anything argparse does not recognise before
    # it is treated the same way, so both of these work:
    #     atlas run optimize -n -- --margin 0.3
    #     atlas run optimize --margin 0.3
    argv = sys.argv[1:]
    if '--' in argv:
        cut = argv.index('--')
        head, tail = argv[:cut], argv[cut + 1:]
    else:
        head, tail = argv, []

    args, unknown = ap.parse_known_args(head)
    args.extra = unknown + tail
    if not getattr(args, 'verb', None):
        ap.print_help()
        print()
        return cmd_list(argparse.Namespace(group=None))
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main() or 0)

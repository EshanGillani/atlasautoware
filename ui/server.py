"""
Race Control dashboard — backend.
=================================

A dependency-free (standard library only) web server.  It is the graphical
front end to exactly the same machinery the `atlas` CLI drives — the command
registry, the environment detection, the hardware doctor — so anything you can
do from a terminal you can do from a laptop or a phone in the pit, and neither
can get out of sync with the other.

    python3 ui/server.py                 # http://127.0.0.1:8000
    python3 ui/server.py --host 0.0.0.0  # reachable from the pit network

**On binding.**  The default is loopback only, deliberately: several endpoints
here start processes that move a real car, and there is no authentication.
Passing `--host 0.0.0.0` opens that to everyone on the network, which is fine
on an isolated pit LAN and a bad idea on conference wifi.  The server says so
at startup rather than leaving you to find out.

Endpoints
    GET  /                        the dashboard
    GET  /api/env                 detected environment (ROS? docker? torch?)
    GET  /api/commands            the command registry, grouped
    GET  /api/doctor              hardware check (runs hw_doctor --json)
    GET  /api/state               live race telemetry
    GET  /api/raceline            current raceline polyline
    GET  /api/tuning              Bayesian tuning history
    GET  /api/sessions            practice sessions
    GET  /api/jobs                running/finished jobs
    GET  /api/jobs/<id>           one job with its output tail
    GET  /api/image/<name>        a PNG from racelines/
    POST /api/run                 {id, args} -> start a registry command
    POST /api/jobs/<id>/stop      stop it
    POST /api/generate            regenerate the raceline
    POST /api/race/{start,stop}   the two-car opponent demo
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))

import atlas                                   # noqa: E402 — environment + build
import atlas_registry as reg                   # noqa: E402

SEED = ('49.910', '42.780')                    # competition-track corridor seed
MAX_TAIL = 400                                 # output lines kept per job
MAX_JOBS = 40                                  # finished jobs kept in memory
MAX_LOG_BYTES = 512 * 1024                     # tail of bayes_log.jsonl to read


# ── job control ──────────────────────────────────────────────────────────────
class Job:
    """One launched command, with a bounded rolling tail of its output."""

    def __init__(self, job_id, cmd_id, argv, shown):
        self.id = job_id
        self.cmd_id = cmd_id
        self.argv = argv
        self.shown = shown
        self.started = time.time()
        self.finished = None
        self.returncode = None
        self.lines = []
        self._lock = threading.Lock()
        # PYTHONUNBUFFERED is essential, not cosmetic: Python block-buffers
        # stdout when it is a pipe rather than a terminal, so a long-running
        # job (a tuning sweep, a training run) shows NOTHING in the dashboard
        # until it exits — which reads as "the button is broken". Forcing
        # line buffering makes progress stream as it happens.
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        self.proc = subprocess.Popen(
            argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors='replace', env=env)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                with self._lock:
                    self.lines.append(line.rstrip('\n'))
                    if len(self.lines) > MAX_TAIL:
                        del self.lines[:len(self.lines) - MAX_TAIL]
        finally:
            # Close the pipe explicitly. Popen holds the read end open until the
            # object is collected, and during a long session that is one OS
            # handle per command ever launched.
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            self.returncode = self.proc.wait()
            self.finished = time.time()

    def running(self):
        return self.finished is None

    def stop(self):
        if self.running():
            try:
                self.proc.terminate()
            except Exception:
                pass

    def as_dict(self, with_output=False):
        d = dict(id=self.id, cmd=self.cmd_id, shown=self.shown,
                 started=self.started, finished=self.finished,
                 running=self.running(), returncode=self.returncode,
                 elapsed=round((self.finished or time.time()) - self.started, 1))
        if with_output:
            with self._lock:
                d['output'] = list(self.lines)
        return d


JOBS = {}
_job_seq = [0]
_jobs_lock = threading.Lock()


def reap_jobs():
    """Drop the oldest FINISHED jobs once we are over the cap.

    Nothing used to leave JOBS, so a dashboard left open through a competition
    day accumulated every command ever launched — each holding its output tail,
    its argv and a Popen object. Running jobs are never evicted, however many
    there are; only completed history is trimmed, newest kept.
    """
    with _jobs_lock:
        done = sorted((j for j in JOBS.values() if not j.running()),
                      key=lambda j: j.finished or 0.0)
        for job in done[:max(0, len(done) - MAX_JOBS)]:
            JOBS.pop(job.id, None)


def launch(cmd_id, extra):
    """Resolve a registry id to a real command and start it."""
    cmd = reg.get(cmd_id)
    if not cmd:
        return None, f'unknown command: {cmd_id}'
    env = atlas.detect()
    ctx, why = atlas.choose_context(cmd, env)
    if ctx is None:
        return None, why
    argv, shown = atlas.build(cmd, extra, env, ctx)
    if ctx == 'docker':
        argv = [a for a in argv if a != '-it']      # no tty behind a web server
    with _jobs_lock:
        _job_seq[0] += 1
        job_id = f'j{_job_seq[0]}'
    try:
        job = Job(job_id, cmd_id, argv, f'[{ctx}] {shown}')
    except Exception as e:
        return None, f'could not start: {e}'
    JOBS[job_id] = job
    reap_jobs()
    return job, None


# ── legacy helpers (raceline studio + the opponent demo) ─────────────────────
def dx(cmd, timeout=180):
    """Run a bash command wherever ROS lives; -> (ok, combined output)."""
    env = atlas.detect()
    if env['native_ros']:
        line = atlas.ros_prefix(env['workspace']) + f'cd {REPO}; ' + cmd
        argv = ['bash', '-lc', line]
    elif env['container']:
        line = (f'source {atlas.ROS_SETUP}; '
                f'source /sim_ws/install/setup.bash 2>/dev/null; '
                f'cd {atlas.CONTAINER_REPO}; ' + cmd)
        argv = ['docker', 'exec', env['container'], 'bash', '-lc', line]
    else:
        return False, 'no ROS available (no local ROS 2, no sim container)'
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, errors='replace')
        return p.returncode == 0, (p.stdout + p.stderr)
    except Exception as e:
        return False, str(e)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    # ── plumbing ────────────────────────────────────────────────────────────
    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass                                   # browser navigated away

    def _body(self):
        n = int(self.headers.get('Content-Length', 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except Exception:
            return {}

    # ── routes ──────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split('?')[0]
        routes = {
            '/': lambda: self._file('index.html', 'text/html; charset=utf-8'),
            '/api/env': self._env,
            '/api/commands': self._commands,
            '/api/doctor': self._doctor,
            '/api/state': self._state,
            '/api/raceline': self._raceline,
            '/api/tuning': self._tuning,
            '/api/sessions': self._sessions,
            '/api/frontier': self._frontier,
            '/api/jobs': self._jobs,
        }
        if path in routes:
            return routes[path]()
        if path.startswith('/api/jobs/'):
            return self._job(path.rsplit('/', 1)[-1])
        if path.startswith('/api/image/'):
            return self._image(path.rsplit('/', 1)[-1])
        return self._send(404, {'error': 'not found'})

    def do_POST(self):
        path = self.path.split('?')[0]
        if path == '/api/run':
            return self._run(self._body())
        if path.startswith('/api/jobs/') and path.endswith('/stop'):
            return self._stop(path.split('/')[3])
        if path == '/api/generate':
            return self._generate(self._body())
        if path == '/api/race/start':
            return self._race_start()
        if path == '/api/race/stop':
            return self._race_stop()
        return self._send(404, {'error': 'not found'})

    # ── implementations ─────────────────────────────────────────────────────
    def _file(self, name, ctype):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if not os.path.exists(p):
            return self._send(404, 'missing ' + name, 'text/plain')
        with open(p, 'rb') as f:
            self._send(200, f.read(), ctype)

    def _image(self, name):
        p = os.path.join(REPO, 'racelines', os.path.basename(name.split('?')[0]))
        if not os.path.exists(p):
            return self._send(404, b'', 'image/png')
        with open(p, 'rb') as f:
            self._send(200, f.read(), 'image/png')

    def _env(self):
        e = atlas.detect()
        e['ros_ready'] = bool(e['native_ros'] or e['container'])
        return self._send(200, e)

    def _commands(self):
        e = atlas.detect()
        out = []
        for gid, label, blurb, cmds in reg.by_group():
            items = []
            for c in cmds:
                ctx, why = atlas.choose_context(c, e)
                items.append(dict(
                    id=c['id'], title=c['title'], long=c.get('long', ''),
                    where=c['where'], env=c['env'],
                    danger=bool(c.get('danger')),
                    args=[dict(flag=f, help=h, default=d)
                          for f, h, d in c.get('args', [])],
                    available=ctx is not None, context=ctx, reason=why))
            if items:
                out.append(dict(id=gid, label=label, blurb=blurb, commands=items))
        return self._send(200, {'groups': out})

    def _doctor(self):
        try:
            p = subprocess.run(
                [sys.executable, os.path.join(REPO, 'tools', 'hw_doctor.py'),
                 '--json'], capture_output=True, text=True, timeout=90,
                cwd=REPO, errors='replace')
            return self._send(200, json.loads(p.stdout))
        except Exception as e:
            return self._send(200, {'checks': [], 'ok': False, 'error': str(e)})

    def _state(self):
        p = os.path.join(REPO, 'runtime', 'race_state.json')
        if not os.path.exists(p):
            return self._send(200, {'running': False})
        try:
            with open(p) as f:
                st = json.load(f)
            st['running'] = (time.time() - st.get('ts', 0)) < 1.5
            return self._send(200, st)
        except Exception:
            return self._send(200, {'running': False})

    def _raceline(self):
        p = os.path.join(REPO, 'racelines', 'best_raceline.csv')
        xs, ys, sp = [], [], []
        try:
            import csv
            with open(p) as f:
                for r in csv.DictReader(f):
                    xs.append(float(r['x']))
                    ys.append(float(r['y']))
                    sp.append(float(r['speed']))
        except Exception:
            pass
        return self._send(200, {'x': xs, 'y': ys, 'speed': sp})

    def _tuning(self):
        """The Bayesian tuning history, best-first."""
        p = os.path.join(REPO, 'runtime', 'bayes_log.jsonl')
        recs = []
        if os.path.exists(p):
            # Read only the tail. bayes_log.jsonl is append-only across every
            # tuning session ever run, so slurping the whole file to then keep
            # the last 200 records makes each request cost more the longer the
            # team has owned the car.
            with open(p, 'rb') as f:
                f.seek(0, os.SEEK_END)
                start = max(0, f.tell() - MAX_LOG_BYTES)
                f.seek(start)
                blob = f.read().decode('utf-8', 'replace')
            lines = blob.splitlines()
            if start:
                lines = lines[1:]              # first line is probably partial
            for line in lines:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
        best = None
        bp = os.path.join(REPO, 'runtime', 'bayes_best.json')
        if os.path.exists(bp):
            try:
                with open(bp) as f:
                    best = json.load(f)
            except Exception:
                pass
        return self._send(200, {'evaluations': recs[-200:], 'best': best})

    def _frontier(self):
        """The lap-time / grip-margin trade curve, if one has been computed.

        Read straight from disk rather than recomputed on request: a frontier
        costs minutes of simulation, so the dashboard shows the last one and
        offers a button to refresh it, instead of blocking a page load.
        """
        p = os.path.join(REPO, 'runtime', 'frontier.json')
        if not os.path.exists(p):
            return self._send(200, {'frontier': [], 'stale': True})
        try:
            with open(p) as f:
                data = json.load(f)
            data['computed'] = os.path.getmtime(p)
            data['stale'] = False
            return self._send(200, data)
        except Exception as e:
            return self._send(200, {'frontier': [], 'error': str(e)})

    def _sessions(self):
        base = os.path.join(REPO, 'practice')
        out = []
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                s = os.path.join(base, name, 'summary.json')
                if os.path.exists(s):
                    try:
                        with open(s) as f:
                            out.append(json.load(f))
                    except Exception:
                        pass
        return self._send(200, {'sessions': out})

    def _jobs(self):
        return self._send(200, {'jobs': [j.as_dict()
                                         for j in reversed(list(JOBS.values()))]})

    def _job(self, job_id):
        job = JOBS.get(job_id)
        if not job:
            return self._send(404, {'error': 'no such job'})
        return self._send(200, job.as_dict(with_output=True))

    def _run(self, body):
        cmd_id = body.get('id', '')
        extra = body.get('args', [])
        if isinstance(extra, str):
            extra = extra.split()
        cmd = reg.get(cmd_id)
        if cmd and cmd.get('danger') and not body.get('confirm'):
            return self._send(400, {'error': 'this command can move a real car '
                                             '— confirm required'})
        job, err = launch(cmd_id, [str(a) for a in extra])
        if err:
            return self._send(400, {'error': err})
        return self._send(200, job.as_dict())

    def _stop(self, job_id):
        job = JOBS.get(job_id)
        if not job:
            return self._send(404, {'error': 'no such job'})
        job.stop()
        return self._send(200, {'ok': True})

    def _generate(self, body):
        m = float(body.get('margin', 0.35))
        a = float(body.get('apex_bias', 1.0))
        al = float(body.get('a_lat', 6.5))
        vm = float(body.get('v_max', 7.0))
        cmd = (f'python3 f1tenth_gym_ros/raceline_optimizer.py '
               f'--map maps/comp_track.yaml --output racelines/best_raceline.csv '
               f'--seed {SEED[0]} {SEED[1]} --margin {m} --apex-bias {a} '
               f'--a-lat {al} --v-max {vm} --no-overlay && '
               f'python3 tools/annotate_raceline.py --image racetrackForComp.png '
               f'--csv racelines/best_raceline.csv --yaml maps/comp_track.yaml '
               f'--out racelines/comp_raceline_annotated.png')
        ok, out = self._optimizer(cmd)
        stats = {}
        for line in out.splitlines():
            if line.startswith(('[speed]', '[centerline]', '[optimize]')):
                stats[line.split(']')[0].strip('[')] = line.split(']', 1)[1].strip()
        return self._send(200, {'ok': ok, 'stats': stats, 'log': out[-1200:],
                                'image': 'comp_raceline_annotated.png?t='
                                         + str(int(time.time()))})

    def _optimizer(self, cmd):
        """The optimizer is plain Python — prefer running it right here rather
        than shelling into a container that may not exist."""
        env = atlas.detect()
        if env['numpy'] and not env['native_ros']:
            try:
                p = subprocess.run(['bash', '-lc', cmd.replace('python3',
                                                               sys.executable)],
                                   cwd=REPO, capture_output=True, text=True,
                                   timeout=300, errors='replace')
                return p.returncode == 0, p.stdout + p.stderr
            except Exception:
                pass                                # fall through to ROS/docker
        return dx(cmd)

    def _race_start(self):
        kill = ("for p in $(ps -eo pid,args | grep -E 'race_agent.py|"
                "opponent_driver.py' | grep -v grep | awk '{print $1}'); "
                "do kill $p 2>/dev/null; done; sleep 1; ")
        reset = (
            "timeout 3 ros2 topic pub --once /initialpose "
            "geometry_msgs/msg/PoseWithCovarianceStamped "
            "'{header: {frame_id: map}, pose: {pose: {position: "
            "{x: 49.815, y: 62.230, z: 0.0}, orientation: {z: -0.9685, w: 0.249}}}}'"
            " >/dev/null 2>&1; "
            "timeout 3 ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "
            "'{header: {frame_id: map}, pose: {position: {x: 45.340, y: 55.192, "
            "z: 0.0}, orientation: {z: -0.9075, w: 0.42}}}' >/dev/null 2>&1; ")
        launch_ = ('nohup python3 f1tenth_gym_ros/opponent_driver.py --cap 3.0 '
                   '>/tmp/opp.log 2>&1 & sleep 1; '
                   'nohup python3 f1tenth_gym_ros/race_agent.py >/tmp/race.log 2>&1 & ')
        ok, out = dx(kill + reset + launch_, timeout=40)
        return self._send(200, {'ok': ok, 'log': out[-600:]})

    def _race_stop(self):
        cmd = ("for p in $(ps -eo pid,args | grep -E 'race_agent.py|"
               "opponent_driver.py' | grep -v grep | awk '{print $1}'); "
               "do kill $p 2>/dev/null; done; "
               "rm -f runtime/race_state.json")
        ok, _ = dx(cmd, timeout=20)
        return self._send(200, {'ok': ok})


def main():
    ap = argparse.ArgumentParser(description='Race Control dashboard.')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1',
                    help='0.0.0.0 to expose on the pit network (see the warning)')
    args = ap.parse_args()

    e = atlas.detect()
    print(f'Race Control  http://{"localhost" if args.host == "127.0.0.1" else args.host}'
          f':{args.port}')
    print(f'  repo        {REPO}')
    print(f'  ROS         ' + ('native' if e['native_ros'] else
                               (f"container {e['container']}" if e['container']
                                else 'unavailable')))
    print(f'  torch       ' + ('yes' if e['torch'] else 'no (RL disabled)'))
    if args.host not in ('127.0.0.1', 'localhost'):
        print('\n  ! Bound to ' + args.host + ' with no authentication, and this '
              'server can start\n    commands that move a real car. Only do this '
              'on a trusted network.\n')
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

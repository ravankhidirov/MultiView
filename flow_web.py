"""MultiFlow: live flow imaging of the pipe cross-section, in the browser.

The probes around one ring (12, 3, 6, 9 o'clock) fire once per frame. Every
A-scan is processed as it arrives (flow_live.py, the method of the
MultiAnalysis2 videos): moving scatter on one physical scale for all probes
(dB re each probe's far-wall echo), beams blocked behind gas, gas where no
probe can see. The page shows the cross-section now, the vertical (12-6) and
horizontal (9-3) planes scrolling in time, activity / gas / bottom-wall
scatter, and the events. Recording works like MultiView (files with names,
Next file = Enter, marks = Space), plus flow_events.csv and a _flow.csv
summary per file.

Sources: the board, a simulation (Demo, with sand bursts), or a recorded
session (Replay, also MultiView recordings).

Run:  python flow_web.py          ->  http://127.0.0.1:8012
      python flow_web.py --demo
"""

import base64
import csv
import json
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# first run: take positions, pipe and calibration from MultiView
if not os.path.exists(os.path.join(HERE, "channels.yaml")):
    src = os.path.join(HERE, "..", "MultiView", "channels.yaml")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(HERE, "channels.yaml"))
        print("[setup] channels.yaml copied from MultiView", flush=True)

import channels as chmod  # noqa: E402
import xsection as xs  # noqa: E402
from channels import NAMES, check_all, frame_limit, load_state, pipe_geometry, public, save_state  # noqa: E402
from flow_live import CODE_GAS, CODE_UNSEEN, DB_HI, DB_LO, EventTracker, FlowLive  # noqa: E402
from multiscan import MultiScan, beside, load_config  # noqa: E402
from recorder import MARK_KINDS, Recorder  # noqa: E402
from sim_flow import SandFakeBoard  # noqa: E402

CFG = load_config()
STATE = load_state(CFG)
DEMO_DEFAULT = "--demo" in sys.argv
FLOW = dict(CFG.get("flow") or {})
VIEW = CFG.get("view") or {}
UPDATE_S = float(VIEW.get("update_s", 0.1))
HISTORY_S = float(VIEW.get("history_s", 60))
CAL_S = float(VIEW.get("calibrate_s", 2.0))
NO_DATA_S = 3.0
QUEUE_MAX = 20000
MAX_COLS_PER_REPLY = 1500


# ---- replay ------------------------------------------------------------------------------

def replay_roots():
    dirs = (CFG.get("replay") or {}).get("directories") or [(CFG.get("recording") or {}).get("directory", "recordings")]
    out = []
    for d in dirs:
        p = d if os.path.isabs(d) else os.path.join(HERE, d)
        p = os.path.abspath(p)
        if p not in out:
            out.append(p)
    return out


def list_sessions():
    """'<folder>/<session>' for every session.json under the replay folders (newest first)."""
    out = []
    for root in replay_roots():
        if not os.path.isdir(root):
            continue
        tag = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == "recordings" else os.path.basename(root)
        for name in sorted(os.listdir(root), reverse=True):
            if os.path.exists(os.path.join(root, name, "session.json")):
                out.append((f"{tag}/{name}", os.path.join(root, name)))
    return out


_LABELS = {}


def session_label(path):
    try:
        key = (path, os.path.getmtime(os.path.join(path, "session.json")))
    except OSError:
        return ""
    if key not in _LABELS:
        _LABELS[key] = _session_label(path)
    return _LABELS[key]


def _session_label(path):
    try:
        s = json.load(open(os.path.join(path, "session.json"), encoding="utf-8"))
        names = [g.get("name") for g in s.get("segments", []) if g.get("name")]
        return ", ".join(names[:3])
    except (OSError, ValueError):
        return ""


class ReplaySource:
    """Plays a recorded session back at `speed` x real time, frame by frame."""

    def __init__(self, folder, speed=1.0):
        with open(os.path.join(folder, "session.json"), encoding="utf-8") as f:
            self.session = json.load(f)
        self.folder, self.speed = folder, float(speed)
        self.rate = float(self.session["board"]["rate"])
        self.channels = [c for c in self.session["channels"] if c.get("mode", "active") == "active"]
        self.segments = []
        for seg in self.session["segments"]:
            arrs = {}
            for c in self.channels:
                fn = seg["files"].get(c["name"])
                if not fn:
                    continue
                path = os.path.join(folder, fn)
                try:
                    with open(path[:-4] + ".json", encoding="utf-8") as fh:
                        n = int(json.load(fh)["samples"])
                except (OSError, ValueError, KeyError):
                    n = int(c.get("points") or 0)          # sidecar missing: samples from session.json
                if n <= 0 or not os.path.exists(path):
                    continue
                m = np.memmap(path, np.int16, "r")
                arrs[c["name"]] = m[: len(m) // n * n].reshape(-1, n)
            if arrs:
                self.segments.append((seg.get("name") or "", arrs))
        self.total = sum(min(len(a) for a in s.values()) for _, s in self.segments)
        self.done = False
        self.segname = ""
        self._gen = self._frames()

    def _frames(self):
        k = 0
        for name, arrs in self.segments:
            self.segname = name
            n = min(len(a) for a in arrs.values())
            for i in range(n):
                for c in self.channels:
                    if c["name"] in arrs:
                        yield k, NAMES.index(c["name"]), np.array(arrs[c["name"]][i])
                k += 1

    def configure_channels(self, voltage, rate, specs):
        self.t0 = time.monotonic()
        self.k = 0

    def recv(self, timeout=1.0):
        if self.done:
            time.sleep(timeout)
            raise TimeoutError("replay finished")
        try:
            k, ch, a = next(self._gen)
        except StopIteration:
            self.done = True
            raise TimeoutError("replay finished")
        due = self.t0 + k / (self.rate * self.speed)
        wait = due - time.monotonic()
        if wait > 0:
            time.sleep(min(wait, timeout))
        self.k = k
        return SimpleNamespace(channel=ch, cycle=k, ascan=a, encoder_a=0, encoder_b=0)

    def scene(self):
        return self.segname

    def close(self):
        pass


# ---- controller ----------------------------------------------------------------------------

class Flow:
    def __init__(self):
        self.lock = threading.RLock()
        self.rec = Recorder(CFG)
        self.running = False
        self.source = "demo" if DEMO_DEFAULT else "live"
        self.error = ""
        self.resolved, self.geo = [], None
        self._q = queue.Queue(maxsize=QUEUE_MAX)
        self._stop = threading.Event()
        self._threads, self._dev = [], None
        self.cal_result = None
        self._cal = None
        self._csv = None
        self.reset()

    def reset(self):
        self.proc = None
        self.events = None
        self.cols = deque()          # (seq, t, v bytes, h bytes, activity, gasf, bottom_db, gas_low)
        self.seq = 0
        self.stats = {}
        self.pc_dropped = 0
        self.replay_info = None
        self.mark_t = {}
        self.latest = None
        self.analysis_skipped = 0
        self._abuf = deque()
        self.t_first = None

    # -- start / stop --
    def start(self, source, session=None, speed=1.0):
        with self.lock:
            if self.running:
                return ""
            self.error = ""
            self.reset()
            try:
                if source == "replay":
                    path = dict(list_sessions()).get(session)
                    if not path:
                        return "pick a recording to replay"
                    src = ReplaySource(path, speed)
                    geo = (src.session.get("board") or {}).get("geometry") or pipe_geometry(STATE["pipe"])
                    res = []
                    for c in src.channels:
                        r = dict(c)
                        if r.get("angle_deg") is None:      # a MultiDAQ session: positions from Setup
                            pos = STATE["channels"].get(r["name"], {}).get("position") or ""
                            r["position"] = pos
                            r["angle_deg"] = (int(pos) % 12) * 30.0 if pos else None
                            r.setdefault("t0_us", geo["wall_us"])
                        res.append(r)
                    rate = src.rate
                    self.replay_info = {"session": session, "frames": src.total, "speed": speed}
                    dev = src
                    dev.configure_channels(0, rate, [])
                else:
                    res, errors, err = check_all(STATE)
                    if errors or err:
                        return "; ".join([f"{k}: {v}" for k, v in errors.items()] + ([err] if err else []))
                    geo = pipe_geometry(STATE["pipe"])
                    rate = float(STATE["board"]["rate"])
                    if source == "demo":
                        dev = SandFakeBoard()
                        dev.setup(res, geo)
                    else:
                        dev = MultiScan(CFG["device"]["ip"], int(CFG["device"].get("port", 10002)))
                    dev.configure_channels(int(STATE["board"]["voltage"]), rate, [r["spec"] for r in res])
                if not [r for r in res if r.get("angle_deg") is not None]:
                    raise ValueError("no probe has a position (Setup)")
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                return self.error
            self.source, self._dev, self.resolved, self.geo, self.rate = source, dev, res, geo, rate
            self.proc = FlowLive(res, geo, FLOW, rate)
            self.events = EventTracker(FLOW, rate)
            self.stats = {r["name"]: {"count": 0, "lost": 0, "last": None, "times": deque(maxlen=int(rate * 3))}
                          for r in res}
            self._q = queue.Queue(maxsize=QUEUE_MAX)
            self._abuf = deque(maxlen=max(10, int(2 * rate)))
            self._stop.clear()
            self._threads = [threading.Thread(target=self._recv_loop, daemon=True),
                             threading.Thread(target=self._work_loop, daemon=True),
                             threading.Thread(target=self._analysis_loop, daemon=True)]
            for t in self._threads:
                t.start()
            self.running = True
            self.t_start = time.time()
            return ""

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=6)
        with self.lock:
            if self.rec.active:
                self._rec_close()
            if self._dev is not None:
                try:
                    self._dev.close()
                except Exception as e:
                    print(f"[flow] close: {e}", flush=True)
                self._dev = None
            self.running = False
            if self.error == "replay finished" or self.error.startswith("no A-scans"):
                self.error = ""
        return ""

    def _recv_loop(self):
        last = time.monotonic()
        while not self._stop.is_set():
            try:
                f = self._dev.recv(timeout=0.5)
            except TimeoutError:
                if isinstance(self._dev, ReplaySource) and self._dev.done:
                    self.error = "replay finished"
                elif time.monotonic() - last > NO_DATA_S:
                    self.error = (f"no A-scans from the board for {time.monotonic() - last:.0f} s "
                                  f"- check the cable, the PC's 192.168.1.x address and the settings")
                continue
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                self._stop.set()
                break
            last = time.monotonic()
            if self.error.startswith("no A-scans"):
                self.error = ""
            try:
                self._q.put_nowait((f, time.time()))
            except queue.Full:
                self.pc_dropped += 1

    def _work_loop(self):
        names = [r["name"] for r in self.resolved]
        pending = {}
        while not (self._stop.is_set() and self._q.empty()):
            try:
                f, t = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            name = NAMES[f.channel] if 0 <= f.channel < len(NAMES) else None
            if name not in self.stats:
                continue
            st = self.stats[name]
            if st["last"] is not None and f.cycle - st["last"] > 1:
                st["lost"] += int(f.cycle - st["last"] - 1)
            st["last"] = f.cycle
            st["count"] += 1
            st["times"].append(t)
            with self.lock:
                if self.rec.active:
                    try:
                        self.rec.write(f, name, t)
                    except OSError as e:
                        self.error = f"recording stopped: {e}"
                        self._rec_close()
            pending.setdefault(f.cycle, {})[name] = f.ascan
            for cyc in sorted(pending):
                fr = pending[cyc]
                if len(fr) == len(names) or cyc < f.cycle - 3:
                    del pending[cyc]
                    if len(fr) == len(names):
                        if len(self._abuf) == self._abuf.maxlen:
                            self.analysis_skipped += 1
                        self._abuf.append((cyc, fr))
                        if self._cal is not None:
                            for n, a in fr.items():
                                self._cal["rows"][n].append(a)
                else:
                    break

    def _analysis_loop(self):
        last = time.monotonic()
        while not (self._stop.is_set() and not self._abuf):
            wait = UPDATE_S - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            batch = []
            while self._abuf:
                batch.append(self._abuf.popleft())
            if batch:
                try:
                    self._process(batch)
                except Exception as e:
                    self.error = f"processing: {type(e).__name__}: {e}"
                    print(f"[flow] {e!r}", flush=True)
            cal = self._cal
            if cal is not None and min(len(v) for v in cal["rows"].values()) >= cal["need"]:
                self._finish_calibration()

    def _process(self, batch):
        cycles = np.array([c for c, _ in batch], dtype=np.float64)
        frames = [fr for _, fr in batch]
        if self.t_first is None:
            self.t_first = cycles[0]
        t = (cycles - self.t_first) / self.rate
        out = self.proc.step(frames)
        if out is None:
            return
        ended = self.events.update(t, out)
        now = time.time()
        with self.lock:
            for i in range(out["m"]):
                gl = out.get("gas_low_mm")
                self.cols.append((self.seq, float(t[i]), out["v"][i].tobytes(), out["h"][i].tobytes(),
                                  float(out["activity"][i]), float(out["gasf"][i]),
                                  float(out["bottom_db"][i]) if np.isfinite(out["bottom_db"][i]) else None,
                                  float(gl[i]) if gl is not None and np.isfinite(gl[i]) else None))
                self.seq += 1
            cut = t[-1] - HISTORY_S
            while self.cols and self.cols[0][1] < cut:
                self.cols.popleft()
            probes = {}
            for n, info in out["probe"].items():
                env, prof = out["env_last"][n]
                p = self.proc.probes[n]
                k = max(1, len(env) // 140)
                probes[n] = dict(info, role=p.role, ref_amp=round(p.ref["amp"]), ref_time=p.ref["time"],
                                 env=[round(float(v), 1) for v in env[::k]],
                                 prof=[round(float(v), 1) for v in prof[::k]], bin_us=0.5 * k, t0=p.t0,
                                 far=p.t0 + p.far_us)
            self.latest = {"t": float(t[-1]), "probes": probes,
                           "scene": self._dev.scene() if hasattr(self._dev, "scene") else ""}
            if self.rec.active:
                self._write_rec(now, t, out, ended)

    # -- re-learn / calibration --
    def relearn(self):
        with self.lock:
            if not self.running:
                return "start first"
            self.proc.relearn()
            return ""

    def calibrate(self):
        with self.lock:
            if not self.running or self.source == "replay":
                return "start the board (or the demo) first; calibration needs live data"
            if self.rec.active:
                return "stop recording first"
            self.cal_result = {"status": "collecting", "started": time.time()}
            self._cal = {"need": int(CAL_S * self.rate), "rows": {r["name"]: [] for r in self.resolved}}
            return ""

    def _finish_calibration(self):
        rows, self._cal = self._cal["rows"], None
        results, errors = {}, {}
        for r in self.resolved:
            a = xs.ProbeAnalyzer(r, self.geo, CFG, self.rate)
            try:
                results[r["name"]] = a.calibrate(np.stack(rows[r["name"]]))
            except ValueError as e:
                errors[r["name"]] = str(e)
        with self.lock:
            when = datetime.now().isoformat(timespec="seconds")
            if not results:
                self.cal_result = {"status": "failed", "errors": errors, "results": {}, "time": when}
                return
            STATE.setdefault("calibration", {}).update(results)
            save_state(STATE)
            self.cal_result = {"status": "done" if not errors else "partly", "results": results,
                               "errors": errors, "time": when}
            res, _, _ = check_all(STATE)
            byname = {r["name"]: r for r in res}
            for r in self.resolved:
                if r["name"] in results and r["name"] in byname:
                    r.update(t0_us=byname[r["name"]]["t0_us"], cal=byname[r["name"]]["cal"],
                             calibrated=True, summary=byname[r["name"]]["summary"])
            # time zero changed: new geometry along the beams, learn the reference again
            self.proc = FlowLive(self.resolved, self.geo, FLOW, self.rate)
        print(f"[cal] ok: {list(results)}; failed: {errors}", flush=True)

    # -- recording --
    def rec_start(self, name, file_name=""):
        with self.lock:
            if not self.running or self.source == "replay":
                return "start the board (or the demo) first"
            try:
                board = dict(STATE["board"], pipe=STATE["pipe"], geometry=self.geo,
                             calibration=STATE.get("calibration") or {}, source=self.source, flow=FLOW)
                self.rec.start(name, self.resolved, board, file_name)
            except (RuntimeError, OSError) as e:
                return str(e)
            self.mark_t = {}
            print(f"[rec] started {self.rec.session['folder']}", flush=True)
            return ""

    def rec_next(self, file_name=""):
        with self.lock:
            self._log_open_event()
            try:
                st = self.rec.next_file(file_name)
            except RuntimeError as e:
                return str(e), None
            self._close_csv()
            return "", st["segment"]

    def rec_stop(self):
        with self.lock:
            if not self.rec.active:
                return "not recording"
            self._rec_close()
            return ""

    def _rec_close(self):
        self._log_open_event()
        self.rec.stop()
        self._close_csv()

    def _log_open_event(self):
        """An event still going on when the file ends is written as it is so far."""
        if self.rec.active and self.events is not None and self.events.cur is not None:
            e = dict(self.events.cur, kind=self.events.cur.get("kind", "") + " (still going)")
            self._log_events([e])

    def _close_csv(self):
        if self._csv is not None:
            self._csv[0].close()
            self._csv = None

    def _write_rec(self, now, t, out, ended):
        seg = self.rec.segment
        folder = self.rec.session["folder"]
        if self._csv is None or self._csv[1] != seg["n"]:
            self._close_csv()
            path = os.path.join(folder, f"{self.rec.session['stamp']}_seg{seg['n']:03d}"
                                + (f"_{seg['name']}" if seg.get("name") else "") + "_flow.csv")
            fh = open(path, "w", newline="", encoding="utf-8")
            w = csv.writer(fh)
            w.writerow(["time", "segment", "segment_s", "activity_z", "gas_fraction", "bottom_wall_db", "gas_low_mm"])
            self._csv = (fh, seg["n"], w)
        fh, _, w = self._csv
        # one summary row per update (0.1 s): maxima over the batch
        gl = out.get("gas_low_mm")
        glv = np.nanmin(gl) if gl is not None and np.isfinite(gl).any() else None
        bd = out["bottom_db"]
        w.writerow([datetime.fromtimestamp(now).isoformat(timespec="milliseconds"), seg["n"],
                    round(now - seg["t0"], 3), round(float(np.max(out["activity"])), 2),
                    round(float(np.max(out["gasf"])), 4),
                    round(float(np.nanmax(bd)), 2) if np.isfinite(bd).any() else "",
                    round(float(glv), 2) if glv is not None else ""])
        if ended:
            self._log_events(ended)

    def _log_events(self, events):
        seg = self.rec.segment
        path = os.path.join(self.rec.session["folder"], "flow_events.csv")
        new = not os.path.exists(path)
        try:
            with open(path, "a", newline="", encoding="utf-8") as f2:
                w2 = csv.writer(f2)
                if new:
                    w2.writerow(["logged", "segment", "file_name", "event", "kind", "start_s", "end_s", "duration_s",
                                 "activity_max", "gas_fraction_max", "bottom_wall_db_max"])
                for e in events:
                    w2.writerow([datetime.now().isoformat(timespec="seconds"), seg["n"], seg.get("name", ""), e["id"],
                                 e.get("kind", ""), round(e["start"], 2), round(e["end"], 2), e.get("duration", ""),
                                 round(e["act_max"], 1), round(e["gas_max"], 4), round(e["bottom_db_max"], 1)])
        except OSError as err:
            print(f"[flow] could not write {path}: {err}", flush=True)

    def mark(self, kind, note, sand_g):
        with self.lock:
            try:
                m = self.rec.mark(kind, note, sand_g)
            except (RuntimeError, ValueError) as e:
                return str(e), None
            self.mark_t[m["id"]] = self.cols[-1][1] if self.cols else 0.0
            return "", {k: v for k, v in m.items() if k != "t"}

    def undo_mark(self):
        with self.lock:
            try:
                m = self.rec.undo_mark()
            except RuntimeError as e:
                return str(e)
            self.mark_t.pop(m["id"], None)
            return ""

    # -- page --
    def columns(self, since):
        with self.lock:
            cols = [c for c in self.cols if c[0] >= since]
            if len(cols) > MAX_COLS_PER_REPLY:
                cols = cols[-MAX_COLS_PER_REPLY:]
            if not cols:
                return {"next": self.seq, "n": 0}
            return {"next": cols[-1][0] + 1, "n": len(cols), "first": cols[0][0],
                    "t": [round(c[1], 3) for c in cols],
                    "v": base64.b64encode(b"".join(c[2] for c in cols)).decode("ascii"),
                    "h": base64.b64encode(b"".join(c[3] for c in cols)).decode("ascii"),
                    "act": [round(c[4], 2) for c in cols], "gasf": [round(c[5], 4) for c in cols],
                    "bot": [None if c[6] is None else round(c[6], 1) for c in cols],
                    "glow": [None if c[7] is None else round(c[7], 1) for c in cols]}

    def status(self):
        with self.lock:
            stats = {}
            for n, s in self.stats.items():
                tt = s["times"]
                rate = (len(tt) - 1) / (tt[-1] - tt[0]) if len(tt) > 1 and tt[-1] > tt[0] else 0.0
                stats[n] = {"count": s["count"], "lost": s["lost"], "rate": round(rate, 1)}
            replay = None
            if self.replay_info and isinstance(self._dev, ReplaySource):
                replay = dict(self.replay_info, frame=self._dev.k, done=self._dev.done)
            return {
                "running": self.running, "source": self.source, "error": self.error,
                "channels": [public(r) if "spec" in r else r for r in self.resolved] if self.running else [],
                "geo": self.geo, "rate": getattr(self, "rate", None),
                "learning": self.proc.learning() if self.proc else 0.0,
                "latest": self.latest, "stats": stats, "pc_dropped": self.pc_dropped,
                "analysis_skipped": self.analysis_skipped,
                "events": self.events.listing() if self.events else [],
                "rec": self.rec.status(), "kinds": list(MARK_KINDS), "marks_t": self.mark_t,
                "calibration": self.cal_result, "replay": replay,
                "flow": FLOW, "codes": {"lo": DB_LO, "hi": DB_HI, "unseen": CODE_UNSEEN, "gas": CODE_GAS},
                "sessions": [(k, session_label(p)) for k, p in list_sessions()[:60]],
                "grid": int(FLOW.get("grid", 128)),
            }


FLOWCTL = Flow()


# ---- setup panel helpers ---------------------------------------------------------------------

def setup_view(st=None):
    st = st or STATE
    out, errors, err = check_all(st)
    res = {}
    for n in NAMES:
        try:
            res[n] = public(chmod.resolve(n, st["channels"][n], st))
        except (ValueError, TypeError, KeyError) as e:
            res[n] = {"error": str(e)}
    try:
        geo = pipe_geometry(st["pipe"])
    except (ValueError, TypeError, KeyError) as e:
        geo = {"error": str(e)}
    return {"board": st["board"], "pipe": st["pipe"], "channels": st["channels"], "resolved": res,
            "errors": errors, "error": err, "geo": geo, "max_rate": frame_limit(st),
            "positions": chmod.POSITIONS, "materials": list(chmod.WALL_MATERIALS),
            "filters": chmod.FILTER_NAMES, "voltages": [100, 200, 400],
            "calibration": st.get("calibration") or {}}


def clean_state(body):
    st = {"board": {"voltage": int(body["board"]["voltage"]), "rate": float(body["board"]["rate"])},
          "pipe": {}, "channels": {}, "calibration": STATE.get("calibration") or {}}
    p = body.get("pipe") or {}
    st["pipe"] = {"diameter_mm": float(p["diameter_mm"]), "diameter_is": "outer" if p.get("diameter_is") == "outer" else "inner",
                  "wall_mm": float(p["wall_mm"]), "wall_material": str(p.get("wall_material") or "steel"),
                  "liquid": "custom" if p.get("liquid") == "custom" else "water",
                  "temperature_c": float(p.get("temperature_c") or 20.0),
                  "custom_speed_m_s": float(p.get("custom_speed_m_s") or 1480.0)}
    for n in NAMES:
        c = (body.get("channels") or {}).get(n) or {}
        rng = c.get("range_mm")
        st["channels"][n] = {"enabled": bool(c.get("enabled")), "label": str(c.get("label") or ""),
                             "position": str(c.get("position") or ""),
                             "frequency": float(c.get("frequency") or 5.0), "gain": float(c.get("gain") or 0),
                             "filter": c.get("filter") or "auto",
                             "range_mm": None if rng in (None, "", "auto") else float(rng)}
    return st


def apply_setup(body):
    global STATE
    if FLOWCTL.rec.active:
        return "stop recording before changing the setup"
    try:
        st = clean_state(body)
    except (ValueError, TypeError, KeyError) as e:
        return f"bad value: {e}"
    out, errors, err = check_all(st)
    if errors or err:
        return "not saved: " + "; ".join([f"{k}: {v}" for k, v in errors.items()] + ([err] if err else []))
    old = STATE
    keep = {}
    for n, c in (old.get("calibration") or {}).items():
        o, nw = old["channels"].get(n, {}), st["channels"].get(n, {})
        if o.get("frequency") == nw.get("frequency") and old["pipe"] == st["pipe"]:
            keep[n] = c
    st["calibration"] = keep
    was, src = FLOWCTL.running, FLOWCTL.source
    if was:
        FLOWCTL.stop()
    STATE = st
    save_state(st)
    if was and src != "replay":
        err = FLOWCTL.start(src)
        if err:
            return "saved, but it did not restart: " + err
    return ""


# ---- http ----------------------------------------------------------------------------------

PAGE_PATH = os.path.join(HERE, "flow_page.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, obj, ctype="application/json"):
        data = (obj if isinstance(obj, str) else json.dumps(_clean(obj), default=_json_default)).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, open(PAGE_PATH, encoding="utf-8").read(), "text/html; charset=utf-8")
        elif url.path == "/api/status":
            self._send(200, FLOWCTL.status())
        elif url.path == "/api/flow":
            q = parse_qs(url.query)
            try:
                since = int(q.get("since", ["0"])[0])
            except ValueError:
                since = 0
            self._send(200, FLOWCTL.columns(since))
        elif url.path == "/api/setup":
            self._send(200, setup_view())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self._body()
        except ValueError:
            body = {}
        p = url.path
        if p == "/api/start":
            self._send(200, {"error": FLOWCTL.start(body.get("source") or "live", body.get("session"),
                                                    float(body.get("speed") or 1.0))})
        elif p == "/api/stop":
            self._send(200, {"error": FLOWCTL.stop()})
        elif p == "/api/relearn":
            self._send(200, {"error": FLOWCTL.relearn()})
        elif p == "/api/calibrate":
            self._send(200, {"error": FLOWCTL.calibrate()})
        elif p == "/api/rec/start":
            self._send(200, {"error": FLOWCTL.rec_start(str(body.get("name") or ""), str(body.get("file_name") or ""))})
        elif p == "/api/rec/next":
            err, seg = FLOWCTL.rec_next(str(body.get("file_name") or ""))
            self._send(200, {"error": err, "segment": seg})
        elif p == "/api/rec/stop":
            self._send(200, {"error": FLOWCTL.rec_stop()})
        elif p == "/api/mark":
            err, m = FLOWCTL.mark(str(body.get("kind") or "mark"), str(body.get("note") or ""), body.get("sand_g"))
            self._send(200, {"error": err, "mark": m})
        elif p == "/api/mark/undo":
            self._send(200, {"error": FLOWCTL.undo_mark()})
        elif p == "/api/check":
            try:
                self._send(200, setup_view(clean_state(body)))
            except (ValueError, TypeError, KeyError) as e:
                self._send(200, {"error": f"bad value: {e}"})
        elif p == "/api/setup":
            self._send(200, {"error": apply_setup(body)})
        else:
            self._send(404, {"error": "not found"})


def _clean(o):
    """JSON for the browser: NaN / infinity become null (JSON has no such numbers)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, deque)):
        return [_clean(v) for v in o]
    return o


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


def main():
    host, port = CFG["web"]["host"], int(CFG["web"]["port"])
    print(f"MultiFlow: http://{host}:{port}" + ("  (demo: simulated flow with sand bursts)" if DEMO_DEFAULT else ""), flush=True)
    try:
        g = pipe_geometry(STATE["pipe"])
        print(f"  bore {g['inner_mm']:.1f} mm, liquid {g['c']:.0f} m/s, far wall at {g['far_us']:.1f} µs", flush=True)
    except ValueError as e:
        print(f"  pipe: {e}", flush=True)
    out, errors, err = check_all(STATE)
    for r in out:
        print(f"  {r['name']}: {r['summary']}", flush=True)
    for k, v in errors.items():
        print(f"  {k}: {v}", flush=True)
    if err:
        print(f"  {err}", flush=True)
    print(f"  replay folders: {', '.join(replay_roots())}", flush=True)
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping (closing files)...", flush=True)
        FLOWCTL.stop()
        print("stopped", flush=True)


if __name__ == "__main__":
    main()

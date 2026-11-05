"""Recording sessions, files and marks for MultiDAQ.

A SESSION runs from "Start recording" to "Stop recording" and gets its own
folder:

    recordings/20261029_103000_<name>/
        20261029_103000_seg001_CH1_5MHz.bin     raw int16 A-scans, one per row
        20261029_103000_seg001_CH1_5MHz.json    sidecar: samples, rate, settings...
        20261029_103000_seg001_CH1_5MHz.idx     per A-scan: frame counter + PC time
        20261029_103000_seg001_CH2_passive.bin  ... one set per channel
        20261029_103000_seg002_CH1_5MHz.bin     "Next file" starts segment 2
        marks.csv                               every mark of the session
        session.json                            channels, segments, marks

A SEGMENT is one file per channel. "Next file" (Enter) closes the current
segment and opens the next one between two A-scans, so nothing is lost or
duplicated between files. A segment can be given a name (e.g. sand0.5g);
it goes into its file names: 20261029_103000_seg002_sand0.5g_CH1_5MHz.bin

The .bin/.json pair is the same format as MultiAlert / MultiChannel
recordings, so the analysis scripts read it as it is:
    np.fromfile(bin, np.int16).reshape(-1, json["samples"])
The .idx file is a numpy structured array, dtype [("cycle", "<i8"), ("t", "<f8")]:
    cycle = the board's frame counter (unwrapped), the same for every channel
            of one frame, so channels can be lined up; a jump > 1 = lost frames
    t     = PC time (unix seconds) when the A-scan arrived

A MARK stores the time and, for every channel, the index of the A-scan in the
current segment's file at that moment (marks.csv column <CH>_ascan), so a
mark points at an exact row of each .bin file.
"""

import csv
import json
import os
import re
import shutil
import time
from datetime import datetime

import numpy as np

from multiscan import beside

FULL_SCALE = 8192.0
IDX_DTYPE = np.dtype([("cycle", "<i8"), ("t", "<f8")])
MARK_KINDS = ("injection start", "injection end", "mark")


def _safe(name):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("._")
    return name[:60]


class _ChannelFile:
    def __init__(self, path, r, flush_every):
        self.path, self.r = path, r
        self.bin = open(path, "wb")
        self.idx = open(path[:-4] + ".idx", "wb")
        self.flush_every = flush_every
        self.buf, self.ibuf = [], []
        self.count = 0
        self.lost = 0
        self.last_cycle = None
        self.samples = None
        self.bytes = 0

    def write(self, frame, t):
        a = np.asarray(frame.ascan, dtype=np.int16)
        if self.samples is None:
            self.samples = len(a)
        elif len(a) != self.samples:
            return                           # a stray A-scan of another setup
        if self.last_cycle is not None and frame.cycle - self.last_cycle > 1:
            self.lost += int(frame.cycle - self.last_cycle - 1)
        self.last_cycle = frame.cycle
        self.buf.append(a)
        self.ibuf.append((frame.cycle, t))
        self.count += 1
        self.bytes += a.nbytes
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if self.buf:
            self.bin.write(np.stack(self.buf).tobytes())
            self.idx.write(np.array(self.ibuf, dtype=IDX_DTYPE).tobytes())
            self.buf, self.ibuf = [], []

    def close(self):
        self.flush()
        self.bin.close()
        self.idx.close()


class Recorder:
    def __init__(self, cfg):
        rc = cfg.get("recording") or {}
        self.root = os.path.abspath(beside(rc.get("directory", "recordings")))
        self.flush_every = int(rc.get("flush_every", 64))
        self.cfg = cfg
        self.active = False
        self.session = None      # dict, see start()
        self.segment = None      # dict of the open segment
        self.files = {}          # channel name -> _ChannelFile of the open segment
        self.marks = []
        self.segments = []       # closed and open segments of this session
        self.last_session = None

    # ---- session ----------------------------------------------------------

    def start(self, name, resolved, board, file_name=""):
        if self.active:
            raise RuntimeError("already recording")
        os.makedirs(self.root, exist_ok=True)
        now = time.time()
        stamp = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
        label = _safe(name)
        folder = os.path.join(self.root, stamp + (f"_{label}" if label else ""))
        os.makedirs(folder, exist_ok=False)
        self.session = {"name": name or "", "stamp": stamp, "folder": folder, "t0": now, "bytes": 0,
                        "board": dict(board),
                        "channels": [{k: v for k, v in r.items() if k != "spec"} for r in resolved]}
        self.resolved = resolved
        self.marks, self.segments = [], []
        self.active = True
        self._open_segment(now, file_name)
        self._write_session()
        return self.status()

    def stop(self):
        if not self.active:
            return self.status()
        now = time.time()
        self._close_segment(now)
        self.session["t1"] = now
        self.active = False
        self._write_session()
        self.last_session = self.session["folder"]
        return self.status()

    def next_file(self, file_name=""):
        if not self.active:
            raise RuntimeError("not recording")
        now = time.time()
        self._close_segment(now)
        self._open_segment(now, file_name)
        self._write_session()
        return self.status()

    # ---- segments -----------------------------------------------------------

    def _open_segment(self, now, file_name=""):
        n = len(self.segments) + 1
        label = _safe(file_name)
        self.segment = {"n": n, "t0": now, "start": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
                        "session_s": round(now - self.session["t0"], 3), "name": label, "files": {}}
        self.files = {}
        for r in self.resolved:
            stem = (f"{self.session['stamp']}_seg{n:03d}" + (f"_{label}" if label else "")
                    + f"_{r['name']}_{r['tag']}")
            path = os.path.join(self.session["folder"], stem + ".bin")
            self.files[r["name"]] = _ChannelFile(path, r, self.flush_every)
            self.segment["files"][r["name"]] = os.path.basename(path)
            self._write_sidecar(self.files[r["name"]], open_=True)
        self.segments.append(self.segment)

    def _close_segment(self, now):
        seg = self.segment
        for name, f in self.files.items():
            f.close()
            self._write_sidecar(f, open_=False, t1=now)
        seg["t1"] = now
        seg["end"] = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
        seg["duration_s"] = round(now - seg["t0"], 3)
        seg["ascans"] = {n: f.count for n, f in self.files.items()}
        seg["lost_frames"] = {n: f.lost for n, f in self.files.items()}
        self.session["bytes"] += sum(f.bytes for f in self.files.values())
        self.files = {}

    def _write_sidecar(self, f, open_, t1=None):
        r, seg = f.r, self.segment
        meta = {
            "device": "Doppler Multiscan (parallel UT)",
            "program": "MultiDAQ",
            "connector": r["name"],
            "socket": r["socket"],
            "mode": r["mode"],
            "label": r["label"],
            "samples": f.samples if f.samples is not None else r["points"],
            "ascans": None if open_ else f.count,
            "dtype": "int16",
            "full_scale": FULL_SCALE,
            "sample_rate_mhz": r["sample_rate_mhz"],
            "rate_hz": float(self.session["board"]["rate"]),
            "pulser": {"voltage": int(self.session["board"]["voltage"]),
                       "frequency": r.get("frequency")},
            "receiver": {"gain": r["gain"], "rectification": "none",
                         "filter_min": r["filter"][0], "filter_max": r["filter"][1]},
            "channel": {k: v for k, v in r.items() if k != "spec"},
            "session": os.path.basename(self.session["folder"]),
            "segment": seg["n"],
            "file_name": seg.get("name", ""),
            "start": seg["start"],
            "end": None if open_ else datetime.fromtimestamp(t1).isoformat(timespec="milliseconds"),
            "lost_frames": None if open_ else f.lost,
            "index_file": os.path.basename(f.path[:-4] + ".idx"),
            "index_dtype": [["cycle", "<i8"], ["t", "<f8"]],
            "marks": [{"id": m["id"], "kind": m["kind"], "ascan": m["ascan"].get(r["name"]),
                       "segment_s": m["segment_s"], "sand_g": m["sand_g"], "note": m["note"]}
                      for m in self.marks if m["segment"] == seg["n"]],
            "recording_complete": not open_,
        }
        if r["mode"] == "active":
            meta.update(range_mm=r["range_mm"], velocity_m_s=r["velocity_m_s"])
        else:
            meta.update(window_ms=r["window_ms"], delay_us=r["delay_us"])
        with open(f.path[:-4] + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    # ---- data -------------------------------------------------------------

    def write(self, frame, name, t):
        f = self.files.get(name)
        if f is not None:
            f.write(frame, t)

    # ---- marks ------------------------------------------------------------

    def mark(self, kind, note="", sand_g=None):
        if not self.active:
            raise RuntimeError("start recording first: a mark belongs to a file")
        if kind not in MARK_KINDS:
            raise ValueError(f"kind must be one of {MARK_KINDS}")
        now = time.time()
        seg = self.segment
        m = {"id": (self.marks[-1]["id"] + 1) if self.marks else 1,
             "time": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
             "t": now,
             "session_s": round(now - self.session["t0"], 3),
             "segment": seg["n"],
             "segment_s": round(now - seg["t0"], 3),
             "kind": kind,
             "sand_g": None if sand_g in (None, "") else float(sand_g),
             "note": str(note or ""),
             "ascan": {n: f.count for n, f in self.files.items()},
             "cycle": {n: f.last_cycle for n, f in self.files.items()}}
        self.marks.append(m)
        self._write_marks()
        return m

    def undo_mark(self):
        if not self.marks:
            raise RuntimeError("no mark to undo")
        m = self.marks.pop()
        self._write_marks()
        return m

    def _write_marks(self):
        names = [r["name"] for r in self.resolved]
        path = os.path.join(self.session["folder"], "marks.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "time", "session_s", "segment", "segment_s", "kind", "sand_g", "note"]
                       + [f"{n}_ascan" for n in names] + [f"{n}_file" for n in names])
            for m in self.marks:
                seg = self.segments[m["segment"] - 1]
                w.writerow([m["id"], m["time"], m["session_s"], m["segment"], m["segment_s"],
                            m["kind"], "" if m["sand_g"] is None else m["sand_g"], m["note"]]
                           + [m["ascan"].get(n, "") for n in names]
                           + [seg["files"].get(n, "") for n in names])
        self._write_session()

    def _write_session(self):
        s = self.session
        out = {"program": "MultiDAQ", "name": s["name"], "folder": os.path.basename(s["folder"]),
               "start": datetime.fromtimestamp(s["t0"]).isoformat(timespec="milliseconds"),
               "end": (datetime.fromtimestamp(s["t1"]).isoformat(timespec="milliseconds")
                       if "t1" in s else None),
               "board": s["board"], "channels": s["channels"],
               "segments": [{k: v for k, v in g.items() if k not in ("t0", "t1")} for g in self.segments],
               "marks": [{k: v for k, v in m.items() if k != "t"} for m in self.marks]}
        tmp = os.path.join(s["folder"], "session.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        os.replace(tmp, os.path.join(s["folder"], "session.json"))

    # ---- for the page -----------------------------------------------------

    def status(self):
        try:
            free = shutil.disk_usage(self.root if os.path.isdir(self.root) else beside(".")).free
        except OSError:
            free = None
        out = {"recording": self.active, "root": self.root, "free_gb": None if free is None else round(free / 1e9, 1),
               "last_session": self.last_session}
        if self.session is None:
            return out
        now = time.time()
        seg = self.segment
        out.update(
            session=os.path.basename(self.session["folder"]),
            folder=self.session["folder"],
            session_s=round((self.session.get("t1") if not self.active else now) - self.session["t0"], 1),
            segment=seg["n"] if seg else None,
            segment_s=round(now - seg["t0"], 1) if (seg and self.active) else None,
            files={n: {"file": os.path.basename(f.path), "ascans": f.count, "mb": round(f.bytes / 1e6, 1),
                       "lost": f.lost} for n, f in self.files.items()},
            segments=[{"n": g["n"], "name": g.get("name", ""), "start": g["start"][11:23], "duration_s": g.get("duration_s"),
                       "ascans": g.get("ascans") or {n: f.count for n, f in self.files.items()},
                       "lost": g.get("lost_frames"), "files": g["files"], "open": "t1" not in g}
                      for g in self.segments[-50:]],
            marks=[{k: v for k, v in m.items() if k != "t"} for m in self.marks[-100:]],
            n_marks=len(self.marks),
            mb_total=round((self.session["bytes"] + sum(f.bytes for f in self.files.values())) / 1e6, 1),
        )
        return out

"""Cross-section of the pipe from pulse-echo probes around one ring.

1. ECHO ANALYSIS, per probe and A-scan (ProbeAnalyzer)
   band-pass around the probe frequency (0.5-1.5 f) -> envelope (Hilbert).
   Time zero t0 = the inner wall surface (calibrated with a full pipe, else
   the round trip through the wall). The wall rings after the pulse, longer
   when gas is behind it; the search starts once that ring-down has decayed
   below the threshold, and runs to just before the far wall:
     - GAS: the first envelope peak above the threshold. Gas reflects almost
       everything, so it is the strongest thing a beam can meet in liquid,
       and nothing behind it is seen. Distance d = c * (t - t0) / 2.
     - FULL: no gas, but the far-wall echo at t0 + 2 x bore / c is there: the
       whole diameter along this beam is liquid.
     - DRY: neither: the sound never left the wall, gas touches the wall here.
     - LOST: the wall ring-down says the wall is wet (calibrated probes only)
       but nothing came back: scattering/attenuation (bubbly flow).
   Threshold = max(k x median envelope in the bore, a fraction of the
   calibrated far-wall amplitude). States and distances are smoothed over a
   few frames (median); a probe whose gas distance jumps around a lot over
   the last 0.5 s is flagged "bubbly".

2. RECONSTRUCTION (reconstruct)
   Each probe gives where gas begins along its beam. Probes on opposite sides
   of one diameter bound the gas on that line from both ends.
     - all lines liquid                    -> full pipe
     - top dry, bottom sees a surface, the probes in between agree with a flat
       surface (dry above it, wet below)     -> stratified, flat surface
     - two perpendicular lines with gas      -> the axis-aligned ellipse through
       the four points where gas begins (exact for 4 points)
     - gas on one line only                  -> an ellipse along that line, its
       width estimated (marked "estimated")
   The gas shape is clipped to the bore; holdup = liquid area / bore area.
   Between the beams the shape is an assumption: it is exact only on the
   beam lines.
"""

import math
from collections import deque

import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

FS_MHZ = 100.0
REF_DECIM = 10            # reference envelope stored in 0.1 us bins
GAS, FULL, DRY, LOST = "gas", "full", "dry", "lost"

DEFAULTS = {
    "blank_us": 1.0,          # nothing is detected before t0 + this, AND before the wall ring-down
                              # has decayed below the threshold (whichever is later)
    "guard_us": 2.0,          # the bore search stops this much before the far wall
    "far_tol_us": 3.0,        # far-wall echo looked for within +- this
    "k_noise": 8.0,           # threshold = k x median envelope in the bore ...
    "cal_frac": 0.2,          # ... or this fraction of the calibrated far-wall echo, the larger
    "min_counts": 40.0,       # ... and never below this (counts)
    "rise_ratio": 2.0,        # an echo must rise this much (x, 2 = 6 dB) above the lowest envelope
                              # of the 1 us before it: a decaying ring-down tail does not
    "smooth_frames": 5,       # median over this many frames
    "bubbly_s": 0.5,          # spread of the gas distance over this long ...
    "bubbly_frac": 0.15,      # ... above this fraction of the bore = "bubbly"
    "dry_ring_ratio": 1.3,    # calibrated: wall ring-down this much above the wet level = dry
    "dry_ring_us": 3.0,       # ... measured from t0 + 0.5 us to t0 + this
    "ref_margin": 2.0,        # calibrated: an echo must exceed the full-pipe envelope x this (6 dB)
    "static_s": 1.0,          # uncalibrated: the ring-down end from the median envelope of this long
    "static_quiet_us": 4.0,   # ... where it stays below the threshold for this long (real ring-downs
                              # come in bursts with gaps; calibrate to see films closer than that)
    "display_points": 300,
}


def settings(cfg):
    s = dict(DEFAULTS)
    s.update((cfg or {}).get("detection") or {})
    return s


class ProbeAnalyzer:
    def __init__(self, r, geo, cfg=None, rate_hz=120.0):
        s = settings(cfg)
        self.s, self.r, self.geo = s, r, geo
        self.c = geo["c"]                          # m/s
        self.bore = geo["inner_mm"]
        self.t0 = float(r["t0_us"])
        f = float(r["frequency"])
        nyq = FS_MHZ / 2
        self.sos = butter(4, [0.5 * f / nyq, min(1.5 * f, 0.95 * nyq) / nyq], btype="bandpass", output="sos")
        self.t_far = self.t0 + geo["far_us"]
        self.n = min(int(r["points"]), int(math.ceil((self.t_far + s["far_tol_us"] + 4) * FS_MHZ)))
        us = lambda t: max(0, min(self.n, int(round(t * FS_MHZ))))
        self.us = us
        self.i0 = us(self.t0 + s["blank_us"])
        self.i1 = max(self.i0 + 10, us(self.t_far - s["guard_us"]))
        self.f0, self.f1 = us(self.t_far - s["far_tol_us"]), us(self.t_far + s["far_tol_us"])
        self.r0, self.r1 = us(self.t0 + 0.5), max(us(self.t0 + 0.5) + 5, us(self.t0 + s["dry_ring_us"]))
        cal = r.get("cal") or {}
        self.cal_far = float(cal.get("far_amp", 0) or 0)
        self.cal_ring = float(cal.get("ring", 0) or 0)
        # calibrated: the full-pipe envelope (wall ring-down, probe near field)
        # becomes part of the threshold, so an echo must stand out from it
        ref = cal.get("ref_env")
        self.ref = None
        if ref:
            ref = np.repeat(np.asarray(ref, dtype=np.float64), REF_DECIM)[:self.n]
            self.ref = np.pad(ref, (0, max(0, self.n - len(ref))), constant_values=0.0)
        k = max(1, int(s["smooth_frames"]))
        self.h_state = deque(maxlen=k)
        self.h_d = deque(maxlen=k)
        self.h_bub = deque(maxlen=max(3, int(s["bubbly_s"] * rate_hz)))
        self.h_env = deque(maxlen=max(10, int(s["static_s"] * rate_hz)))   # for the uncalibrated ring end
        self.static_end = self.i0
        self.last_env = None

    def envelope(self, block):
        x = np.asarray(block, dtype=np.float64)[:, :self.n]
        if x.shape[1] < self.n:
            x = np.pad(x, ((0, 0), (0, self.n - x.shape[1])))
        y = sosfiltfilt(self.sos, x, axis=1)
        return np.abs(hilbert(y, N=next_fast_len(self.n), axis=1))[:, :self.n]

    def _static_ring_end(self, thr):
        """Uncalibrated: where the probe's own ring-down ends, from the median
        envelope of the last static_s (the ring is there in every A-scan; a
        steady echo further in only starts after a quiet gap)."""
        E = np.median(np.stack(self.h_env), axis=0)            # 0.1 us bins
        q = max(1, int(self.s["static_quiet_us"] * FS_MHZ / REF_DECIM))
        a = self.i0 // REF_DECIM
        b = self.i1 // REF_DECIM
        seg = E[a:b]
        if len(seg) <= q:
            return self.i0
        ahead = maximum_filter1d(seg, size=q, origin=-(q // 2), mode="nearest")
        quiet = np.flatnonzero(ahead < thr)
        return self.i0 if not quiet.size else (a + int(quiet[0])) * REF_DECIM

    def analyze(self, block):
        """block: (m, samples) A-scans of this probe, oldest first.
        Returns per-A-scan raw results (lists) and updates the smoothed state."""
        s = self.s
        env = self.envelope(block)
        W = env[:, self.i0:self.i1]
        noise = np.median(W, axis=1)
        thr = np.maximum(s["k_noise"] * noise, s["min_counts"])
        if self.cal_far > 0:
            thr = np.maximum(thr, s["cal_frac"] * self.cal_far)
        cols = np.arange(W.shape[1])[None, :]
        # where the search starts
        if self.ref is not None:
            prof = s["ref_margin"] * self.ref[self.i0:self.i1][None, :]
            start = np.zeros(len(env), dtype=int)
        else:
            for e in env:
                self.h_env.append(np.maximum.reduceat(e, np.arange(0, self.n, REF_DECIM)))
            self.static_end = self._static_ring_end(float(np.median(thr)))
            prof = 0.0
            start = np.full(len(env), max(0, self.static_end - self.i0))
        # ... and not before this A-scan's own ring-down has decayed (0.5 us quiet)
        quiet = int(0.5 * FS_MHZ)
        ahead = maximum_filter1d(W, size=quiet, axis=1, origin=-(quiet // 2), mode="nearest")
        below = ahead < np.maximum(thr[:, None], prof)
        ring_end = np.where(below.any(axis=1), np.argmax(below, axis=1), W.shape[1])
        start = np.maximum(start, ring_end)
        # an echo ARRIVES: above the threshold, above the full-pipe reference
        # (calibrated), and well above the envelope of the microsecond before
        # it (a ring-down tail or ripple only decays)
        back = int(1.0 * FS_MHZ)
        before = minimum_filter1d(W, size=back, axis=1, origin=(back - 1) // 2, mode="nearest")
        above = ((W > thr[:, None]) & (W > prof) & (W >= s["rise_ratio"] * before)
                 & (cols >= start[:, None]))
        has = above.any(axis=1)
        first = np.argmax(above, axis=1)
        far_amp = env[:, self.f0:self.f1].max(axis=1) if self.f1 > self.f0 else np.zeros(len(env))
        far_ok = far_amp > np.maximum(thr, 0.3 * self.cal_far if self.cal_far else 0)
        ring = np.sqrt(np.mean(env[:, self.r0:self.r1] ** 2, axis=1))
        dry_ring = (ring >= s["dry_ring_ratio"] * self.cal_ring) if self.cal_ring > 0 else np.zeros(len(env), bool)
        out = []
        win = int(2.0 * FS_MHZ)                     # peak search after the crossing
        for i in range(len(env)):
            if dry_ring[i] and not far_ok[i]:
                st, d, amp = DRY, 0.0, 0.0           # the wall rings on: gas right behind it
            elif has[i]:
                j0 = int(first[i])
                seg = W[i, j0:j0 + win]
                j = j0 + int(np.argmax(seg))
                t = (self.i0 + j) / FS_MHZ
                d = self.c * (t - self.t0) / 2000.0          # mm
                st, amp = GAS, float(W[i, j])
            elif far_ok[i]:
                st, d, amp = FULL, None, float(far_amp[i])
            else:
                wet = self.cal_ring > 0 and not dry_ring[i]
                st, d, amp = (LOST if wet else DRY), (None if wet else 0.0), 0.0
            out.append((st, d, amp, float(thr[i]), float(ring[i]), float(far_amp[i])))
            self._push(st, d)
        self.last_env = env[-1]
        self.last_thr = float(thr[-1])
        self.last_prof = (np.maximum(prof[0], thr[-1]) if self.ref is not None else None)
        return out

    def _push(self, st, d):
        self.h_state.append(st)
        self.h_d.append(d if st == GAS else None)
        self.h_bub.append(d if st == GAS else None)

    def state(self):
        """Smoothed: majority state of the last frames, median distance."""
        if not self.h_state:
            return {"state": LOST, "d": None, "bubbly": False}
        sts = list(self.h_state)
        st = max(set(sts), key=lambda k: (sts.count(k), sts[::-1].index(k) * -1))
        ds = [d for d in self.h_d if d is not None]
        d = float(np.median(ds)) if (st == GAS and ds) else (0.0 if st == DRY else None)
        bd = [x for x in self.h_bub if x is not None]
        bubbly = False
        if len(bd) >= 5:
            q75, q25 = np.percentile(bd, [75, 25])
            bubbly = (q75 - q25) > self.s["bubbly_frac"] * self.bore
        return {"state": st, "d": d, "bubbly": bool(bubbly)}

    def display(self):
        """Envelope of the newest A-scan from t0 to past the far wall, for the
        beam colours and the A-scan plot, with where the search starts and
        the threshold."""
        if self.last_env is None:
            return None
        a = int(max(0, round(self.t0 * FS_MHZ)))
        b = min(len(self.last_env), self.f1 + int(2 * FS_MHZ))
        e = self.last_env[a:b]
        n = int(self.s["display_points"])
        edges = np.linspace(0, len(e), min(n, len(e)) + 1).astype(int)
        out = {"env": [round(float(v), 1) for v in np.maximum.reduceat(e, edges[:-1])],
               "t_from": round(a / FS_MHZ, 3), "t_to": round(b / FS_MHZ, 3), "t0": round(self.t0, 3),
               "search_from": round((self.static_end if self.ref is None else self.i0) / FS_MHZ, 3),
               "far": round(self.t_far, 3), "thr": round(self.last_thr, 1),
               "calibrated": self.ref is not None}
        if self.last_prof is not None:
            p = np.full(len(self.last_env), self.last_thr)
            p[self.i0:self.i1] = self.last_prof
            pe = p[a:b]
            out["thr_prof"] = [round(float(v), 1) for v in np.maximum.reduceat(pe, edges[:-1])]
        return out

    def calibrate(self, block):
        """Pipe full of liquid, no gas: the strongest echo within +-20 % of
        the expected far-wall time is the far wall. Gives time zero, the
        far-wall amplitude, the wall ring-down level and the full-pipe
        envelope (the reference the threshold follows). Raises ValueError."""
        env = self.envelope(block)
        m = env.mean(axis=0)
        far_us = self.geo["far_us"]
        lo = int(max(0, (self.t_far - 0.2 * far_us) * FS_MHZ))
        hi = int(min(env.shape[1], (self.t_far + 0.2 * far_us) * FS_MHZ))
        if hi - lo < 10:
            raise ValueError("the depth window is too short for the far wall")
        seg = env[:, lo:hi]
        amps = seg.max(axis=1)
        # the far-wall echo is followed by its own reverberations, which can
        # be stronger than its start: time it like a gas echo, the peak within
        # 2 us of where it first reaches half its maximum
        onset = np.argmax(seg >= 0.5 * amps[:, None], axis=1)
        w = int(2.0 * FS_MHZ)
        peaks = np.array([o + int(np.argmax(row[o:o + w])) for row, o in zip(seg, onset)])
        t_far = (lo + float(np.median(peaks))) / FS_MHZ
        amp = float(np.median(amps))
        a = int(max(0, (self.t0 + self.s["blank_us"]) * FS_MHZ))
        bore = m[a:lo]
        noise = float(np.percentile(bore, 20)) if bore.size else 0.0
        if amp < max(8 * noise, self.s["min_counts"]) or np.std(peaks) / FS_MHZ > 1.0:
            raise ValueError(f"no steady far-wall echo near {self.t_far:.1f} µs "
                             f"(found {amp:.0f} counts vs noise {noise:.0f}): is the pipe full "
                             f"of liquid without bubbles, the probe coupled, and are the diameter "
                             f"and the temperature right?")
        # past the probe's ring-down, nothing in the bore may come close to the far wall
        q = int(2.0 * FS_MHZ)
        ahead = maximum_filter1d(bore, size=q, origin=-(q // 2), mode="nearest") if bore.size > q else bore
        quiet = np.flatnonzero(ahead < 8 * noise)
        ring_end = a + (int(quiet[0]) if quiet.size else len(bore))
        if ring_end < lo and m[ring_end:lo].max() > 0.5 * amp:
            j = ring_end + int(np.argmax(m[ring_end:lo]))
            raise ValueError(f"a strong echo at {j / FS_MHZ:.1f} µs, before the far wall: gas in the pipe?")
        t0 = t_far - far_us
        r0, r1 = int(max(0, (t0 + 0.5) * FS_MHZ)), int(max(0, (t0 + self.s["dry_ring_us"]) * FS_MHZ))
        ring = float(np.median(np.sqrt(np.mean(env[:, r0:max(r1, r0 + 5)] ** 2, axis=1))))
        # the full-pipe envelope up to just before the far wall, 0.1 us bins;
        # the 90th percentile over the A-scans so normal variation stays below it
        p90 = np.percentile(env, 90, axis=0)[:lo]
        ref = np.maximum.reduceat(p90, np.arange(0, len(p90), REF_DECIM))
        return {"t0_us": round(t0, 3), "far_us_measured": round(t_far, 3), "far_amp": round(amp, 1),
                "ring": round(ring, 1), "noise": round(noise, 1),
                "ring_end_us": round(ring_end / FS_MHZ, 2),
                "ref_env": [int(round(v)) for v in ref]}


# ---- reconstruction ---------------------------------------------------------------------

def _unit(angle_deg):
    a = math.radians(angle_deg)
    return math.sin(a), math.cos(a)            # x right, y up; 0 deg = 12 o'clock


def reconstruct(probes, R):
    """probes: [{"name", "angle", "state", "d", "bubbly"}] (smoothed, with a
    position). R: bore radius, mm. Returns shape, holdup, regime, lines."""
    P = [p for p in probes if p.get("angle") is not None]
    for p in P:
        p["u"] = _unit(p["angle"])
        p["y"] = R * p["u"][1]
        if p["state"] == GAS and p["d"] is not None:
            p["d"] = float(min(max(p["d"], 0.0), 2 * R))
    lost = [p for p in P if p["state"] == LOST or p.get("bubbly")]
    notes = []

    # gas interval on each diameter: s along the direction of the first probe
    lines = {}
    for p in P:
        key = round(p["angle"] % 180.0, 1)
        lines.setdefault(key, []).append(p)
    L = []
    for key, ps in lines.items():
        ux, uy = _unit(key)
        hi = lo = None
        known_hi = known_lo = False
        # A far-wall echo is unmistakable: that whole diameter is liquid, and
        # a "gas" echo from the other end of it was a bubble or a particle.
        nogas = any(p["state"] == FULL for p in ps)
        for p in ps:
            if nogas:
                if p["state"] == GAS:
                    notes.append(f"{p['name']}: echo at {p['d']:.1f} mm, but the opposite probe "
                                 f"sees the far wall: a bubble or particle, not a gas boundary")
                continue
            same = abs(((p["angle"] - key) + 180) % 360 - 180) < 1
            if p["state"] == GAS:
                v = R - p["d"]
            elif p["state"] == DRY:
                v = R
            else:
                continue
            if same:
                hi, known_hi = v, True
            else:
                lo, known_lo = -v, True
        if known_hi and known_lo and lo >= hi:
            # each end saw gas beyond where the other one did: one gas body
            # spans the line between them anyway, keep the outer limits
            lo, hi = min(lo, hi), max(lo, hi)
        L.append({"angle": key, "u": (ux, uy), "nogas": nogas,
                  "lo": lo, "hi": hi, "known_lo": known_lo, "known_hi": known_hi})

    top = max(P, key=lambda p: p["y"]) if P else None
    bot = min(P, key=lambda p: p["y"]) if P else None
    shape = {"type": "unknown"}
    regime = "unclear"
    est = False

    gas_lines = [l for l in L if (l["known_lo"] or l["known_hi"]) and not l["nogas"]]
    if P and not gas_lines and all(l["nogas"] for l in L):
        shape, regime = {"type": "full"}, "full liquid"
    elif top is not None and bot is not None and top is not bot and top["state"] == DRY \
            and bot["state"] == GAS and _flat_ok(P, R, -R + bot["d"], top, bot):
        level = -R + bot["d"]
        shape, regime = {"type": "stratified", "level": level}, "stratified"
    elif top is not None and top["state"] == DRY and bot is not None and bot["state"] == DRY \
            and all(p["state"] == DRY for p in P):
        shape, regime = {"type": "empty"}, "no liquid at the probes (dry)"
    else:
        both = [l for l in gas_lines if l["known_lo"] and l["known_hi"]]
        if len(both) >= 2:
            l1, l2 = _most_perpendicular(both)
            ell = _ellipse_through(l1, l2)
            if ell is not None:
                shape = dict(ell, type="ellipse")
            else:
                shape = dict(_ellipse_from_line(l1, R), type="ellipse")
                est = True
        elif gas_lines:
            l1 = max(gas_lines, key=lambda l: (l["known_lo"] + l["known_hi"]))
            shape = dict(_ellipse_from_line(l1, R), type="ellipse")
            est = True
        if shape["type"] == "ellipse":
            wet_all = all(p["state"] == GAS and p["d"] > 0.5 for p in P)
            touches_top = top is not None and top["state"] == DRY
            regime = ("gas core (annular)" if wet_all else
                      "elongated bubble / gas at the top" if touches_top else "gas pocket")
            if est:
                regime += " (width estimated)"
    if len(lost) >= max(2, len(P) // 2):
        regime = "bubbly / no clear interface"
    holdup = _holdup(shape, R)
    return {"shape": shape, "holdup": holdup, "regime": regime, "estimated": est,
            "notes": notes, "lost": [p["name"] for p in lost]}


def _flat_ok(P, R, level, top, bot, tol_frac=0.08):
    """Probes between top and bottom agree with a flat surface at `level`."""
    tol = tol_frac * 2 * R
    for p in P:
        if p is top or p is bot:
            continue
        if p["y"] > level + tol and p["state"] not in (DRY,):
            return False
        if p["y"] < level - tol and p["state"] == DRY:
            return False
    return True


def _most_perpendicular(lines):
    best, pair = -1, (lines[0], lines[1])
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a = abs(math.sin(math.radians(lines[i]["angle"] - lines[j]["angle"])))
            if a > best:
                best, pair = a, (lines[i], lines[j])
    return pair


def _ellipse_through(l1, l2):
    """Ellipse with axes along l1 and l2 (assumed perpendicular) through the
    4 points where gas begins on them. In the frame (s1 along l1, s2 along
    l2): points (lo1,0), (hi1,0), (0,lo2), (0,hi2).
    (s1-c1)^2/a^2 + (s2-c2)^2/b^2 = 1, solved exactly."""
    c1, c2 = (l1["lo"] + l1["hi"]) / 2, (l2["lo"] + l2["hi"]) / 2
    p1, p2 = ((l1["hi"] - l1["lo"]) / 2) ** 2, ((l2["hi"] - l2["lo"]) / 2) ** 2
    # c2^2 A + p1 B ... with A = 1/b^2 (along l2), B = 1/a^2 (along l1):
    # points on l1 (s2 = 0): (s1-c1)^2 B + c2^2 A = 1 -> p1 B + c2^2 A = 1
    # points on l2 (s1 = 0): c1^2 B + p2 A = 1
    M = np.array([[p1, c2 ** 2], [c1 ** 2, p2]])
    try:
        B, A = np.linalg.solve(M, [1.0, 1.0])
    except np.linalg.LinAlgError:
        return None
    if A <= 0 or B <= 0:
        return None
    a, b = 1 / math.sqrt(B), 1 / math.sqrt(A)
    u1, u2 = l1["u"], l2["u"]
    cx = c1 * u1[0] + c2 * u2[0]
    cy = c1 * u1[1] + c2 * u2[1]
    rot = math.degrees(math.atan2(u1[1], u1[0]))      # direction of axis a
    return {"cx": cx, "cy": cy, "a": a, "b": b, "rot_deg": rot}


def _ellipse_from_line(l, R):
    """Gas known on one line only: centred on that line, width = its length
    (capped by the bore chord there)."""
    lo = l["lo"] if l["known_lo"] else (-l["hi"] if l["known_hi"] else 0)
    hi = l["hi"] if l["known_hi"] else -lo
    c = (lo + hi) / 2
    a = max((hi - lo) / 2, 0.5)
    chord = math.sqrt(max(R * R - c * c, 1.0))
    b = min(a, chord)
    u = l["u"]
    return {"cx": c * u[0], "cy": c * u[1], "a": a, "b": b,
            "rot_deg": math.degrees(math.atan2(u[1], u[0]))}


_GRID = {}


def gas_mask(shape, X, Y):
    t = shape.get("type")
    if t in ("full", "unknown"):
        return np.zeros_like(X, dtype=bool)
    if t == "empty":
        return np.ones_like(X, dtype=bool)
    if t == "stratified":
        return Y > shape["level"]
    if t == "ellipse":
        th = math.radians(shape["rot_deg"])
        dx, dy = X - shape["cx"], Y - shape["cy"]
        s1 = dx * math.cos(th) + dy * math.sin(th)
        s2 = -dx * math.sin(th) + dy * math.cos(th)
        return (s1 / shape["a"]) ** 2 + (s2 / shape["b"]) ** 2 <= 1
    if t == "circles":                    # used by the simulator only
        m = np.zeros_like(X, dtype=bool)
        for (cx, cy, r) in shape["circles"]:
            m |= (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        return m
    if t == "union":
        m = np.zeros_like(X, dtype=bool)
        for s in shape["parts"]:
            m |= gas_mask(s, X, Y)
        return m
    raise ValueError(f"unknown shape {t}")


def _holdup(shape, R, n=160):
    if shape.get("type") == "unknown":
        return None
    key = (round(R, 3), n)
    if key not in _GRID:
        g = (np.arange(n) + 0.5) / n * 2 * R - R
        X, Y = np.meshgrid(g, -g)
        _GRID[key] = (X, Y, X ** 2 + Y ** 2 <= R * R)
    X, Y, inside = _GRID[key]
    gas = gas_mask(shape, X, Y) & inside
    return round(1.0 - gas.sum() / inside.sum(), 4)


def holdup_of(shape, R):
    return _holdup(shape, R)

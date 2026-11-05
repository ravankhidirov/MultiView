"""
Real-time flow imaging along the probe beams (the method of
MultiAnalysis2/flow_export.py, made causal: only past A-scans are used).

Per probe and per A-scan, in 0.5 us depth bins:
  moving power   power of the A-scan-to-A-scan change in the sand band
                 (1.2-2.0 x probe frequency): sand, bubbles - anything moving
  envelope       main band (0.6-1.5 x probe frequency), for new echoes
Against the probe's REFERENCE (learned for a few seconds of normal flow, then
slowly following drift while nothing happens):
  scatter_db     (moving power - reference moving power) / far-wall echo^2,
                 in dB: one physical scale for every probe, gain and coupling
                 cancel
  echo_db        envelope over the reference envelope, dB
Beams: a probe stops counting behind the first new echo >= shadow_db; gas on
its own wall (wall ringing up >= ring_db) blinds it; only 1.5 mm .. 2 mm short
of the far wall is used. The two probes of a plane are averaged where both
see; where neither sees because of gas (echo >= gas_db or gas on the wall),
the point is gas.
Per frame: a column of the vertical (12-6) and horizontal (9-3) planes,
the gas fraction along the beams, the scatter near the bottom wall and an
activity value (how much of the depth range changes, z, strongest probe).
"""
import time
from collections import deque

import numpy as np
from scipy import fft as sfft

FS = 100.0          # MHz
BIN_US = 0.5
DEC = 5             # bands are mixed down to complex 20 MHz signals (0.05 us)
ROLES = ("top", "right", "bottom", "left")
CODE_SEEN_MAX = 250  # codes 0..250 = dB in [DB_LO, DB_HI]
CODE_UNSEEN = 252
CODE_GAS = 255
DB_LO, DB_HI = -60.0, -20.0


def role(angle_deg):
    a = float(angle_deg) % 360
    return ROLES[int(((a + 45) % 360) // 90)]


def _causal_mean(tail, new, k):
    """Running mean over the last k rows; `tail` holds the previous k-1 rows."""
    x = np.vstack([tail, new]) if tail is not None and len(tail) else new
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1]), x.dtype), x]), axis=0)
    n = len(new)
    hi = np.arange(len(x) - n + 1, len(x) + 1)
    lo = np.maximum(hi - k, 0)
    out = (c[hi] - c[lo]) / (hi - lo)[:, None]
    return out, x[-(k - 1):] if k > 1 else x[:0]


class Probe:
    def __init__(self, r, geo, cfg):
        self.name = r["name"]
        self.angle = float(r["angle_deg"])
        self.role = role(self.angle)
        self.t0 = float(r["t0_us"])
        f = float(r["frequency"])
        self.R = float(geo["radius_mm"])
        self.c = float(geo["c"]) / 1000.0                     # mm/us
        self.far_us = float(geo["far_us"])
        self.n = None
        self.f = f
        self.cfg = cfg
        self.prev_hi = None
        self.tail_pw = self.tail_env = self.tail_act = None
        self.ref = None
        self.learn = None
        self.adapt_alpha = 0.0

    def _setup(self, n_raw):
        self.n_raw = n_raw
        # only up to a little past the far wall is needed; a multiple of 50 samples
        n = min(n_raw, int((self.t0 + self.far_us + 6.0) * FS))
        n = max(50, (n // 50) * 50)
        self.n = n
        self.L = n // DEC
        self.bw = int(BIN_US * FS) // DEC                    # baseband samples per 0.5 us bin
        self.nb = n // int(BIN_US * FS)
        fr = np.fft.rfftfreq(n, 1 / FS)

        def band(lo, hi):
            k = np.where((fr >= lo) & (fr <= hi))[0]
            return k[0], k[-1] + 1
        self.khi = band(1.2 * self.f, min(2.0 * self.f, 0.49 * FS))
        self.klo = band(0.6 * self.f, 1.5 * self.f)
        t = (np.arange(self.nb) + 0.5) * BIN_US
        self.d = (t - self.t0) * self.c / 2.0                 # mm from this probe's inner wall
        self.inside = (self.d >= 0) & (self.d <= 2 * self.R)
        self.inb = np.where(self.inside & (self.d >= 1.5) & (self.d <= 2 * self.R - 2.0))[0]
        b0 = int(self.t0 / BIN_US)
        self.ring = slice(b0 + 1, b0 + 12)
        fb = int((self.t0 + self.far_us) / BIN_US); fh = int(2.5 / BIN_US)
        self.far = slice(max(0, fb - fh), min(self.nb, fb + fh + 1))
        self.act_bins = slice(3, min(self.nb, 120))
        bot = self.inside & (self.d >= 1.5) & (self.d <= 13.0)
        self.near_bins = np.where(bot)[0]

    def _baseband(self, F, k):
        """Analytic signal of band F[k0:k1], mixed down, at 20 MHz (complex64)."""
        k0, k1 = k
        m = F.shape[0]
        Z = np.zeros((m, self.L), np.complex64)
        Z[:, : k1 - k0] = F[:, k0:k1]
        return sfft.ifft(Z, axis=1, workers=-1) * np.float32(2.0 * self.L / self.n)

    # -- per batch --
    def features(self, X):
        """X: (m, n) int16 A-scans -> per-frame moving power, envelope, log moving rms (per bin)."""
        if self.n is None or getattr(self, "n_raw", None) != X.shape[1]:
            self._setup(X.shape[1])
            self.prev_hi = None
            self.tail_pw = self.tail_env = self.tail_act = None
        X = np.asarray(X[:, : self.n], np.float32)
        X = X - X.mean(1, keepdims=True)
        m = X.shape[0]
        nb, bw = self.nb, self.bw
        F = sfft.rfft(X, axis=1, workers=-1)
        ah = self._baseband(F, self.khi)[:, : nb * bw]
        prev = self.prev_hi if self.prev_hi is not None else ah[:1]
        dh = np.diff(np.vstack([prev, ah]), axis=0)
        self.prev_hi = ah[-1:]
        pw = 0.5 * (np.abs(dh) ** 2).reshape(m, nb, bw).mean(2)       # power of the real band signal's change
        al = self._baseband(F, self.klo)[:, : nb * bw]
        env = np.abs(al).reshape(m, nb, bw).max(2)
        k = int(self.cfg["smooth_frames"])
        pw_s, self.tail_pw = _causal_mean(self.tail_pw, pw, k)
        env_s, self.tail_env = _causal_mean(self.tail_env, env, max(1, k // 2))
        lhd = np.log(np.sqrt(pw) + 1.0)
        act_s, self.tail_act = _causal_mean(self.tail_act, lhd, int(self.cfg["activity_frames"]))
        return pw_s, env_s, act_s

    def start_learning(self, frames):
        self.learn = {"need": int(frames), "pw": [], "env": [], "act": []}

    def learn_step(self, pw, env, act):
        L = self.learn
        L["pw"].append(pw); L["env"].append(env); L["act"].append(act)
        got = sum(len(a) for a in L["pw"])
        if got >= L["need"]:
            P, E, A = (np.vstack(L[k]) for k in ("pw", "env", "act"))
            self.learn = None
            med = np.median(A, 0)
            self.ref = {"pq": np.median(P, 0) + 1e-9, "prof": np.median(E, 0) + 1e-6,
                        "amp": float(max(np.median(E[:, self.far].max(1)), 1.0)),
                        "act_med": med, "act_mad": np.maximum(np.median(np.abs(A - med), 0) * 1.4826, 1e-3),
                        "time": time.strftime("%H:%M:%S"), "frames": len(P)}
        return got

    def adapt(self, pw, env, quiet_mask, alpha):
        """Follow slow drift: frames with nothing going on nudge the reference."""
        if self.ref is None or alpha <= 0 or not quiet_mask.any():
            return
        k = float(quiet_mask.sum())
        a = 1 - (1 - alpha) ** k
        r = self.ref
        r["pq"] = np.exp((1 - a) * np.log(r["pq"]) + a * np.log(np.median(pw[quiet_mask], 0) + 1e-9))
        r["prof"] = np.exp((1 - a) * np.log(r["prof"]) + a * np.log(np.median(env[quiet_mask], 0) + 1e-6))

    def evaluate(self, pw, env, act):
        """Physical level, echo, shadows, activity for a batch (needs the reference)."""
        r, cfg = self.ref, self.cfg
        lin = np.clip(pw - r["pq"], 0, None) / r["amp"] ** 2
        e = 20 * np.log10(env / r["prof"] + 1e-9)
        ring = 20 * np.log10(env[:, self.ring].mean(1) / r["prof"][self.ring].mean())
        blocked = ring >= cfg["ring_db"]
        dd = self.d[self.inb]

        def first_at(thr):
            hit = e[:, self.inb] >= thr
            return np.where(hit.any(1), dd[np.argmax(hit, 1)], np.inf)
        shadow = np.where(blocked, 0.0, first_at(cfg["shadow_db"]))
        gas_from = np.where(blocked, 0.0, first_at(cfg["gas_db"]))
        z = (act[:, self.act_bins] - r["act_med"][self.act_bins]) / r["act_mad"][self.act_bins]
        activity = np.percentile(z, 90, axis=1)
        return {"lin": lin, "e": e, "shadow": shadow, "gas_from": gas_from, "activity": activity,
                "ring_db": ring, "env": env}


class Plane:
    """The two probes of one plane on a common line through the centre."""

    def __init__(self, probes, axis, R, n_grid):
        self.probes, self.axis, self.R = probes, axis, R
        self.s = np.linspace(-R, R, n_grid)
        self._interp = {}

    def _weights(self, p):
        key = (p.name, p.nb)
        if key not in self._interp:
            sign = 1.0 if p.role in ("top", "right") else -1.0
            dist = self.R - sign * self.s
            bins = np.where(p.inside)[0]
            k = np.interp(dist, p.d[bins], np.arange(len(bins)))
            k0 = np.clip(np.floor(k).astype(int), 0, len(bins) - 2)
            f = (k - k0).astype(np.float32)
            usable = (dist >= 1.5) & (dist <= 2 * self.R - 2.0)
            self._interp[key] = (dist, bins[k0], bins[k0 + 1], f, usable)
        return self._interp[key]

    def combine(self, ev):
        tot = cnt = gas_any = None
        for p in self.probes:
            if p.name not in ev:
                continue
            q = ev[p.name]
            dist, i0, i1, f, usable = self._weights(p)
            lin = q["lin"][:, i0] * (1 - f) + q["lin"][:, i1] * f
            sees = usable[None, :] & (dist[None, :] < (q["shadow"][:, None] - 1.0))
            wall_gas = (q["gas_from"][:, None] == 0) & (dist[None, :] < 1.5)
            behind = ((dist[None, :] >= q["gas_from"][:, None]) & usable[None, :]) | wall_gas
            if tot is None:
                tot = np.zeros(lin.shape, np.float64); cnt = np.zeros(lin.shape, np.int8)
                gas_any = np.zeros(lin.shape, bool)
            tot += np.where(sees, lin, 0); cnt += sees; gas_any |= behind
        if tot is None:
            return None
        db = 10 * np.log10(tot / np.maximum(cnt, 1) + 10 ** (DB_LO / 10))
        gas = (cnt == 0) & gas_any
        code = np.clip(np.round((db - DB_LO) / (DB_HI - DB_LO) * CODE_SEEN_MAX), 0, CODE_SEEN_MAX).astype(np.uint8)
        code[cnt == 0] = CODE_UNSEEN
        code[gas] = CODE_GAS
        return {"code": code, "db": np.where(cnt > 0, db, np.nan), "gas": gas, "cnt": cnt}


class FlowLive:
    """Feed batches of frames ({probe name: A-scan}) with step(); read the
    per-frame results it returns."""

    def __init__(self, resolved, geo, cfg, rate):
        self.cfg = cfg
        self.rate = float(rate)
        self.R = float(geo["radius_mm"])
        self.probes = {}
        for r in resolved:
            if r.get("angle_deg") is None:
                continue
            p = Probe(r, geo, cfg)
            self.probes[p.name] = p
        byrole = {p.role: p for p in self.probes.values()}
        self.byrole = byrole
        g = int(cfg["grid"])
        self.vplane = Plane([byrole[k] for k in ("top", "bottom") if k in byrole], "y", self.R, g)
        self.hplane = Plane([byrole[k] for k in ("left", "right") if k in byrole], "x", self.R, g)
        tau = float(cfg.get("adapt_tau_s") or 0)
        self.alpha = 1.0 / (tau * self.rate) if tau > 0 else 0.0
        self.relearn()

    def relearn(self):
        for p in self.probes.values():
            p.ref = None
            p.start_learning(int(float(self.cfg["learn_s"]) * self.rate))

    def learning(self):
        """Fraction learned (1 = ready)."""
        out = []
        for p in self.probes.values():
            if p.ref is not None:
                out.append(1.0)
            elif p.learn is not None:
                out.append(min(1.0, sum(len(a) for a in p.learn["pw"]) / max(p.learn["need"], 1)))
            else:
                out.append(0.0)
        return min(out) if out else 0.0

    def step(self, frames):
        """frames: list of {name: ascan}. Returns per-frame results (dict of arrays) or None while learning."""
        names = [n for n in self.probes if all(n in fr for fr in frames)]
        if not names or not frames:
            return None
        ev, feats = {}, {}
        for n in names:
            p = self.probes[n]
            pw, env, act = p.features(np.stack([fr[n] for fr in frames]))
            feats[n] = (pw, env, act)
            if p.ref is None:
                if p.learn is None:
                    p.start_learning(int(float(self.cfg["learn_s"]) * self.rate))
                p.learn_step(pw, env, act)
                continue
            ev[n] = p.evaluate(pw, env, act)
        if len(ev) < len(names):
            return None
        m = len(frames)
        activity = np.max(np.vstack([ev[n]["activity"] for n in names]), 0)
        v = self.vplane.combine(ev)
        h = self.hplane.combine(ev)
        out = {"m": m, "activity": activity}
        G = self.vplane.s
        for key, pl in (("v", v), ("h", h)):
            if pl is None:
                out[key] = np.full((m, len(G)), CODE_UNSEEN, np.uint8)
                out[key + "_gasf"] = np.zeros(m)
                continue
            out[key] = pl["code"]
            out[key + "_gasf"] = pl["gas"].mean(1)
            if key == "v":
                # lowest gas on the vertical line -> liquid height under the gas (mm from the bottom)
                anyg = pl["gas"].any(1)
                low = np.where(anyg, G[np.argmax(pl["gas"], 1)] + self.R, np.nan)
                out["gas_low_mm"] = low
        b = self.byrole.get("bottom")
        if b is not None and b.name in ev:
            q = ev[b.name]
            nb = b.near_bins
            sees = b.d[nb][None, :] < (q["shadow"][:, None] - 1.0)
            lin = np.where(sees, q["lin"][:, nb], np.nan)
            with np.errstate(all="ignore"):
                out["bottom_db"] = 10 * np.log10(np.nanpercentile(lin, 90, axis=1) + 10 ** (DB_LO / 10))
        else:
            out["bottom_db"] = np.full(m, np.nan)
        out["gasf"] = np.maximum(out["v_gasf"], out["h_gasf"])
        # slow reference drift: frames where nothing happens
        quiet = (activity < float(self.cfg["event_z"]) * 0.5) & (out["gasf"] == 0)
        for n in names:
            pw, env, _ = feats[n]
            self.probes[n].adapt(pw, env, quiet, self.alpha)
        out["probe"] = {n: {"ring_db": float(ev[n]["ring_db"][-1]), "shadow_mm": float(ev[n]["shadow"][-1]),
                            "gas_from_mm": float(ev[n]["gas_from"][-1]),
                            "activity": float(ev[n]["activity"][-1])} for n in names}
        out["env_last"] = {n: (feats[n][1][-1], self.probes[n].ref["prof"]) for n in names}
        return out


class EventTracker:
    """Activity above event_z for >= event_min_s starts an event; below it for event_end_s ends it.
    Each event: duration, peak activity, peak gas fraction, peak bottom-wall scatter
    and what it looked like: gas (gas seen along the beams), sand-like (scatter near
    the bottom wall >= sand_db), both, or just activity."""

    def __init__(self, cfg, rate):
        self.cfg, self.rate = cfg, float(rate)
        self.events = deque(maxlen=200)
        self.cur = None
        self.above = 0
        self.below = 0
        self.n = 0

    def kind(self, e):
        g = e["gas_max"] >= float(self.cfg["gas_frac_min"])
        s = e["bottom_db_max"] >= float(self.cfg["sand_db"])
        return "sand-like + gas" if (g and s) else "gas" if g else "sand-like" if s else "activity"

    def update(self, t, out):
        """t: per-frame time (s since start). Returns events that just ended."""
        done = []
        thr = float(self.cfg["event_z"])
        need = int(float(self.cfg["event_min_s"]) * self.rate)
        endq = int(float(self.cfg["event_end_s"]) * self.rate)
        for i in range(out["m"]):
            a = float(out["activity"][i]); g = float(out["gasf"][i]); b = out["bottom_db"][i]
            b = float(b) if np.isfinite(b) else -99.0
            active = a > thr                      # gas only describes an event (steady gas layers are no event)
            if self.cur is None:
                self.above = self.above + 1 if active else 0
                if self.above >= need:
                    self.n += 1
                    self.cur = {"id": self.n, "start": float(t[i]) - need / self.rate, "end": float(t[i]),
                                "act_max": a, "gas_max": g, "bottom_db_max": b, "open": True}
                    self.below = 0
            else:
                c = self.cur
                c["end"] = float(t[i])
                c["act_max"] = max(c["act_max"], a); c["gas_max"] = max(c["gas_max"], g)
                c["bottom_db_max"] = max(c["bottom_db_max"], b)
                self.below = 0 if active else self.below + 1
                if self.below >= endq:
                    c["end"] = float(t[i]) - endq / self.rate
                    c["open"] = False
                    c["kind"] = self.kind(c)
                    c["duration"] = round(c["end"] - c["start"], 2)
                    self.events.append(c)
                    done.append(c)
                    self.cur = None
                    self.above = 0
        if self.cur is not None:
            self.cur["kind"] = self.kind(self.cur)
            self.cur["duration"] = round(self.cur["end"] - self.cur["start"], 2)
        return done

    def listing(self, k=30):
        ev = list(self.events)[-k:]
        if self.cur is not None:
            ev.append(self.cur)
        return ev

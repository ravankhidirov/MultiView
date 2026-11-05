"""Simulated flow and A-scans, for `view_web.py --demo` and for testing.

A scripted sequence of flow regimes in the cross-section (the truth), and
for every probe the A-scan it would record: wall ring-down (longer when gas
touches the wall), the echo of the nearest gas along the beam, the far-wall
echo when the whole diameter is liquid, bubbles as weaker echoes, and noise.
"""

import math

import numpy as np

from xsection import gas_mask

SCENES = [  # (name, seconds)
    ("full liquid", 12), ("stratified, waves", 15), ("stratified, low", 12),
    ("elongated bubble", 12), ("gas core (annular)", 12), ("slug flow", 16),
    ("bubbly", 12),
]
CYCLE = sum(s for _, s in SCENES)


def truth(t, R, rng=None):
    """Gas shape in the bore at time t (s); R: bore radius, mm."""
    t = t % CYCLE
    for name, dur in SCENES:
        if t < dur:
            break
        t -= dur
    D = 2 * R
    if name == "full liquid":
        return name, {"type": "full"}
    if name == "stratified, waves":
        h = 0.55 * D + 0.10 * D * math.sin(2 * math.pi * t / 5) + 0.02 * D * math.sin(2 * math.pi * 1.3 * t)
        return name, {"type": "stratified", "level": -R + h}
    if name == "stratified, low":
        return name, {"type": "stratified", "level": -R + 0.3 * D + 0.03 * D * math.sin(2 * math.pi * t / 3)}
    if name == "elongated bubble":
        return name, {"type": "ellipse", "cx": 0.0, "cy": 0.45 * R, "a": 0.72 * R, "b": 0.62 * R, "rot_deg": 0.0}
    if name == "gas core (annular)":
        return name, {"type": "ellipse", "cx": 0.03 * R * math.sin(t), "cy": 0.12 * R,
                      "a": 0.74 * R, "b": 0.70 * R, "rot_deg": 0.0}
    if name == "slug flow":
        ph = t % 4.0
        if ph < 1.5:
            return name, {"type": "full"}
        return name, {"type": "stratified", "level": -R + 0.4 * D}
    # bubbly: many small bubbles moving through
    r = np.random.default_rng(int(t * 30))
    circles = [(float(r.uniform(-0.8, 0.8) * R), float(r.uniform(-0.8, 0.8) * R), float(r.uniform(0.5, 2.5)))
               for _ in range(40)]
    return name, {"type": "circles", "circles": circles}


def first_gas(shape, R, angle_deg, step=0.1):
    """Distance from the probe (inner wall) along its beam to the first gas:
    0 = gas at the wall, None = none across the whole diameter."""
    ux, uy = math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))
    s = np.arange(0, 2 * R, step)
    X, Y = (R - s) * ux, (R - s) * uy
    m = gas_mask(shape, X, Y) & (X ** 2 + Y ** 2 <= R * R + 1e-9)
    if not m.any():
        return None
    return float(s[np.argmax(m)])


def _burst(n, t_us, f_mhz, amp, width_us=None):
    """A Gaussian-windowed tone burst centred at t_us, on an n-sample 100 MHz axis."""
    w = width_us or 1.5 / f_mhz
    i0, i1 = max(0, int((t_us - 3 * w) * 100)), min(n, int((t_us + 3 * w) * 100) + 1)
    out = np.zeros(n, dtype=np.float32)
    if i1 > i0:
        tt = np.arange(i0, i1) / 100.0 - t_us
        out[i0:i1] = amp * np.exp(-(tt / w) ** 2) * np.sin(2 * np.pi * f_mhz * tt)
    return out


def ascan(shape, R, geo, r, rng, n=None):
    """One probe's A-scan for this gas shape (int16 counts)."""
    n = n or int(r["points"])
    f = float(r["frequency"])
    t0 = geo["wall_us"]                      # true time zero
    d = first_gas(shape, R, r["angle_deg"]) if r.get("angle_deg") is not None else None
    tt = np.arange(n, dtype=np.float32) / 100.0
    dry = d is not None and d < 0.3
    tau = 2.5 if dry else 0.8                # ring-down lasts longer with gas behind the wall
    x = (6000 * np.exp(-tt / tau) * np.sin(2 * np.pi * f * tt)).astype(np.float32)
    x += rng.standard_normal(n).astype(np.float32) * 12
    c = geo["c"]
    if dry:
        pass
    elif d is None:
        x += _burst(n, t0 + geo["far_us"], f, 2500 * (1 + 0.1 * rng.standard_normal()))
    else:
        x += _burst(n, t0 + 2 * d / c * 1e3, f, 3000 * (1 + 0.15 * rng.standard_normal()))
    if shape.get("type") == "circles" and not dry:
        # extra weak scatter along the path in bubbly flow
        stop = d if d is not None else 2 * R
        for s in rng.uniform(1, max(1.5, stop), size=6):
            x += _burst(n, t0 + 2 * s / c * 1e3, f, float(rng.uniform(60, 250)))
    return np.clip(x, -8192, 8191).astype(np.int16), d


class FakeBoard:
    """Stands in for MultiScan in --demo: frames at the frame rate, every
    configured channel once per frame, signals from the scripted flow."""

    def __init__(self, *a, **k):
        self.specs, self.k, self.i = [], 0, 0
        self.rng = np.random.default_rng(3)
        self.resolved, self.geo = [], None

    def setup(self, resolved, geo):
        self.resolved, self.geo = {r["spec"].channel: r for r in resolved}, geo

    def get_version(self, timeout=2.0):
        return (0, 0)

    def configure_channels(self, voltage, rate, specs):
        import time
        self.specs, self.rate = list(specs), float(rate)
        self.t0, self.k, self.i = time.monotonic(), 0, 0
        self._shape = None

    def recv(self, timeout=1.0):
        import time
        from types import SimpleNamespace
        if not self.specs:
            time.sleep(timeout)
            raise TimeoutError
        due = self.t0 + self.k / self.rate
        wait = due - time.monotonic()
        if wait > timeout:
            time.sleep(timeout)
            raise TimeoutError
        if wait > 0:
            time.sleep(wait)
        if self.i == 0:
            self._name, self._shape = truth(self.k / self.rate, self.geo["radius_mm"])
        c = self.specs[self.i]
        r = self.resolved[c.channel]
        a, _ = ascan(self._shape, self.geo["radius_mm"], self.geo, r, self.rng, c.points)
        f = SimpleNamespace(channel=c.channel, cycle=self.k, ascan=a, encoder_a=0, encoder_b=0)
        self.i += 1
        if self.i == len(self.specs):
            self.i, self.k = 0, self.k + 1
        return f

    def scene(self):
        return getattr(self, "_name", "")

    def close(self):
        self.specs = []

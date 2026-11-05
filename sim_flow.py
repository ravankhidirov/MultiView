"""Demo board for MultiFlow: MultiView's simulated flow (sim.py) plus sand.

Every 30 s (from 6 to 9 s into each half minute) a 3 s sand burst is
added: many weak moving scatterers in the bottom ~9 mm of the pipe, a new
random set every frame (they move), each only about as strong as the noise,
as in the real recordings. The bottom probe sees them close to its wall, the
top probe near the far wall (when the path is liquid), the side probes a few.
"""
import math

import numpy as np

from sim import FakeBoard, _burst, first_gas

SAND_EVERY_S, SAND_FOR_S, SAND_AT_S = 30.0, 3.0, 6.0


class SandFakeBoard(FakeBoard):
    def recv(self, timeout=1.0):
        f = super().recv(timeout)
        t = f.cycle / self.rate
        ph = t % SAND_EVERY_S
        if SAND_AT_S <= ph < SAND_AT_S + SAND_FOR_S:
            r = self.resolved.get(f.channel)
            if r is not None and r.get("angle_deg") is not None:
                f.ascan = self._add_sand(f.ascan, r, ph - SAND_AT_S)
        return f

    def _add_sand(self, a, r, tin):
        geo, R = self.geo, self.geo["radius_mm"]
        rng, fq, c = self.rng, float(r["frequency"]), geo["c"]
        t0 = geo["wall_us"]
        strength = math.sin(math.pi * tin / SAND_FOR_S) ** 2           # rises and falls
        ang = float(r["angle_deg"]) % 360
        stop = first_gas(self._shape, R, ang)                         # gas blocks the beam
        stop = 2 * R if stop is None else stop
        if abs(ang - 180) < 1:                                        # bottom probe: close to its wall
            d = rng.uniform(2.0, 9.0, size=int(60 * strength) + 1)
        elif abs(ang) < 1:                                            # top probe: near the far wall
            d = rng.uniform(2 * R - 9.0, 2 * R - 2.0, size=int(60 * strength) + 1)
        else:                                                         # sides: a few through the middle
            d = rng.uniform(0.3 * R, 1.7 * R, size=int(6 * strength))
        x = a.astype(np.float32)
        for s in d[d < stop]:
            x += _burst(len(x), t0 + 2 * s / c * 1e3, fq * 1.4, float(rng.uniform(8, 30)) * strength)
        return np.clip(x, -8192, 8191).astype(np.int16)

    def scene(self):
        t = self.k / self.rate if self.rate else 0
        ph = t % SAND_EVERY_S
        sand = SAND_AT_S <= ph < SAND_AT_S + SAND_FOR_S
        return super().scene() + (" + sand burst" if sand else "")

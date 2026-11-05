"""Probes, pipe and liquid for MultiView.

Each of the 8 channels is off or on; an "on" channel is an active probe
(pulse-echo on socket Tn) at a clock position around ONE cross-section of
the pipe (12 = top, 3 = right seen from upstream, 6 = bottom, 9 = left).
All channels that are on fire once per frame, one after another, so the
four views of the cross-section belong to the same instant (within one frame).

Pipe: diameter (inner or outer, as given), wall thickness and wall
material. Liquid: water at a temperature (speed of sound from it), or a
speed typed in. Calibration (pipe full of liquid, no gas) measures each
probe's time zero from the far-wall echo.

Everything here is saved in channels.yaml beside the program.
"""

import copy
import math
import os

import yaml

from multiscan import (DIGITIZER_MHZ, MAX_GAIN_DB, UT_FILTERS, VOLTAGES, ChannelSpec,
                       beside, build_groups, depth_to_samples, pulse_width_ns, rate_limit)

CHANNELS_FILE = "channels.yaml"
NAMES = [f"CH{i}" for i in range(1, 9)]
FILTER_NAMES = [f"{lo:g}-{hi:g}" for lo, hi in UT_FILTERS]
POSITIONS = ["12", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
WALL_MATERIALS = {          # longitudinal speed of sound, m/s
    "steel": 5920.0, "stainless steel": 5790.0, "aluminium": 6320.0,
    "acrylic (PMMA)": 2730.0, "PVC": 2380.0, "glass": 5640.0,
}

DEFAULT_CHANNEL = {"enabled": False, "label": "", "position": "", "frequency": 5.0,
                   "gain": 60.0, "filter": "auto", "range_mm": None}
DEFAULT_PIPE = {"diameter_mm": 50.8,          # 2 in
                "diameter_is": "outer",       # inner | outer (2 in outside, 2 mm wall -> 46.8 mm bore)
                "wall_mm": 2.0,
                "wall_material": "steel",
                "liquid": "water",            # water | custom
                "temperature_c": 20.0,
                "custom_speed_m_s": 1480.0}


# ---- speed of sound ------------------------------------------------------------------

def water_speed(t_c):
    """Speed of sound in pure water, m/s (Marczak 1997, 0-95 C, +-0.02 %)."""
    t = float(t_c)
    return (1402.385 + 5.038813 * t - 5.799136e-2 * t ** 2 + 3.287156e-4 * t ** 3
            - 1.398845e-6 * t ** 4 + 2.787860e-9 * t ** 5)


def pipe_geometry(pipe):
    """Inner radius, wall, speeds: everything the cross-section needs."""
    d = float(pipe["diameter_mm"])
    wall = float(pipe["wall_mm"])
    if d <= 0 or wall < 0:
        raise ValueError("pipe diameter must be > 0 and wall >= 0 mm")
    inner = d if pipe.get("diameter_is", "inner") == "inner" else d - 2 * wall
    if inner <= 0:
        raise ValueError("the wall is thicker than the pipe")
    mat = pipe.get("wall_material", "steel")
    if mat not in WALL_MATERIALS:
        raise ValueError(f"wall material must be one of {list(WALL_MATERIALS)}")
    if pipe.get("liquid", "water") == "water":
        c = water_speed(pipe.get("temperature_c", 20.0))
    else:
        c = float(pipe["custom_speed_m_s"])
    if not 300 < c < 3000:
        raise ValueError("liquid speed of sound must be 300-3000 m/s")
    c_wall = WALL_MATERIALS[mat]
    return {"inner_mm": inner, "outer_mm": inner + 2 * wall, "radius_mm": inner / 2,
            "wall_mm": wall, "wall_material": mat, "c_wall": c_wall, "c": c,
            "wall_us": 2 * wall / c_wall * 1e3,                  # round trip through the wall
            "far_us": 2 * inner / c * 1e3}                       # round trip across the bore


# ---- channels.yaml -------------------------------------------------------------------

def default_state(cfg):
    chans = {n: dict(DEFAULT_CHANNEL) for n in NAMES}
    for n, pos in zip(NAMES[:4], ["12", "3", "6", "9"]):
        chans[n].update(enabled=True, position=pos)
    b = cfg.get("board") or {}
    return {"board": {"voltage": int(b.get("voltage", 400)), "rate": float(b.get("rate", 120.0))},
            "pipe": dict(DEFAULT_PIPE, **(cfg.get("pipe") or {})),
            "channels": chans, "calibration": {}}


def load_state(cfg):
    st = default_state(cfg)
    path = beside(CHANNELS_FILE)
    if not os.path.exists(path):
        return st
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    st["board"].update(data.get("board") or {})
    st["pipe"].update(data.get("pipe") or {})
    for n in NAMES:
        c = dict(DEFAULT_CHANNEL)
        c.update((data.get("channels") or {}).get(n) or {})
        c["position"] = str(c.get("position") or "")
        st["channels"][n] = c
    st["calibration"] = data.get("calibration") or {}
    return st


def save_state(st):
    path = beside(CHANNELS_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# Written by view_web.py (Setup panel). Safe to edit by hand while the\n"
                "# program is stopped. calibration: written by the Calibrate button.\n")
        yaml.safe_dump({"board": st["board"], "pipe": st["pipe"], "channels": st["channels"],
                        "calibration": st.get("calibration") or {}}, f,
                       sort_keys=False, default_flow_style=False)
    os.replace(tmp, path)


# ---- resolving -------------------------------------------------------------------------

def pick_filter(lo, hi, min_coverage=0.8):
    span = hi - lo
    scored = []
    for flo, fhi in UT_FILTERS[1:]:
        cover = max(0.0, min(hi, fhi) - max(lo, flo)) / span
        scored.append((cover, -(fhi - flo), (flo, fhi)))
    good = [s for s in scored if s[0] >= min_coverage]
    if good:
        return max(good, key=lambda s: (s[1], s[0]))[2]
    best = max(scored)
    return best[2] if best[0] >= 0.5 else UT_FILTERS[0]


def resolve(name, ch, st):
    """One probe as it will run, with its geometry. ValueError if impossible."""
    idx = NAMES.index(name)
    geo = pipe_geometry(st["pipe"])
    f = float(ch["frequency"])
    width = pulse_width_ns(f)
    gain = float(ch["gain"])
    if not 0 <= gain <= MAX_GAIN_DB:
        raise ValueError(f"gain must be 0-{MAX_GAIN_DB:g} dB")
    if ch.get("filter") in (None, "", "auto"):
        flt = pick_filter(0.5 * f, 1.5 * f)
    elif ch["filter"] in FILTER_NAMES:
        flt = UT_FILTERS[FILTER_NAMES.index(ch["filter"])]
    else:
        raise ValueError(f"board filter must be 'auto' or one of {FILTER_NAMES}")
    pos = str(ch.get("position") or "")
    if pos and pos not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}")
    # depth window: across the bore and some way past the far wall (for the
    # far-wall echo), in mm of liquid
    rng = ch.get("range_mm")
    if rng in (None, "", "auto"):
        rng = math.ceil((geo["inner_mm"] * 1.25 + 5) / 5) * 5
    rng = float(rng)
    points = depth_to_samples(rng, geo["c"], 1)
    window_ns = points * 10
    if window_ns / 1000 < geo["far_us"] + geo["wall_us"] + 3:
        raise ValueError(f"depth window {rng:g} mm ends before the far wall "
                         f"({geo['inner_mm']:.1f} mm bore): make it larger or 'auto'")
    spec = ChannelSpec(idx, True, True, width, gain, tuple(flt), "none", 0, window_ns, points)
    cal = (st.get("calibration") or {}).get(name) or {}
    angle = (int(pos) % 12) * 30.0 if pos else None
    r = {"name": name, "label": ch.get("label", ""), "mode": "active", "position": pos,
         "angle_deg": angle, "frequency": f, "pulse_ns": width, "gain": gain,
         "filter": list(flt), "range_mm": rng, "velocity_m_s": geo["c"],
         "sample_rate_mhz": float(DIGITIZER_MHZ), "points": points, "window_us": window_ns / 1000,
         "socket": f"T{idx + 1}", "tag": f"{f:g}MHz".replace(".", "p"),
         "t0_us": float(cal["t0_us"]) if "t0_us" in cal else round(geo["wall_us"], 3),
         "calibrated": "t0_us" in cal, "cal": cal, "spec": spec}
    pos_txt = f"{pos} o'clock" if pos else "no position"
    r["summary"] = (f"socket T{idx + 1} · {pos_txt} · {f:g} MHz · "
                    f"{gain:g} dB · filter {flt[0]:g}-{flt[1]:g} MHz · window {rng:g} mm = "
                    f"{window_ns / 1000:.1f} µs · time zero {r['t0_us']:.2f} µs"
                    + (" (calibrated)" if r["calibrated"] else " (from the wall, not calibrated)"))
    return r


def check_all(st, cfg=None):
    """(resolved list, errors by channel, general error)."""
    errors, out = {}, []
    try:
        pipe_geometry(st["pipe"])
    except (ValueError, TypeError, KeyError) as e:
        return [], {}, f"pipe: {e}"
    for n in NAMES:
        ch = st["channels"][n]
        if not ch.get("enabled"):
            continue
        try:
            out.append(resolve(n, ch, st))
        except (ValueError, TypeError, KeyError) as e:
            errors[n] = str(e)
    err = ""
    positions = [r["position"] for r in out if r["position"]]
    if len(positions) != len(set(positions)):
        err = "two channels have the same position"
    try:
        voltage, rate = int(st["board"]["voltage"]), float(st["board"]["rate"])
        if voltage not in VOLTAGES:
            raise ValueError(f"voltage must be one of {VOLTAGES}")
        if out and not errors:
            build_groups(voltage, rate, [r["spec"] for r in out])
    except (ValueError, TypeError, KeyError) as e:
        err = err or str(e)
    if not out and not errors:
        err = err or "no channel is switched on"
    return out, errors, err


def frame_limit(st):
    out, errors, _ = check_all(dict(st, board=dict(st["board"], rate=1.0)))
    if not out or errors:
        return None
    try:
        return rate_limit(int(st["board"]["voltage"]), [r["spec"] for r in out])
    except ValueError:
        return None


def public(r):
    return {k: v for k, v in r.items() if k != "spec"}

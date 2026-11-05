"""Client for the Doppler Multiscan multi-channel UT board, over its own SDK.

The board speaks UDP only. Parameters go out through `DplBoard.dll` (the
vendor's C++ SDK, called here with ctypes); data comes back through a callback
the DLL fires once per 1024-byte packet: a 24-byte header, then up to 500
big-endian int16 samples. An A-scan longer than 500 points arrives as several
packets, numbered by the header's package id, and is reassembled here.

This board is the *parallel* Multiscan (EBoardType Multiscan_Par_UT): its 8
channels are set up in groups through SendParallelUtGroup. Channels of one
group fire together, groups fire one after another. Each connector here gets
a group of its own, so two probes on the same pipe never hear each other's
pulse. `acquisition.rate` is the frame PRF - how many times per second the
whole sequence runs - so every channel delivers that many A-scans.

Verified on the rig (FPGA 0x4b034bc1 at 192.168.1.130): typed as the serial
Multiscan_Ser_UT the board takes the parameters and sends nothing.
"""

import ctypes as C
import json
import math
import os
import queue
import re
import socket
import time
from collections import namedtuple

import numpy as np
import yaml

DEFAULT_IP = "192.168.1.130"   # factory address of a Multiscan board
DEFAULT_PORT = 10002

# EBoardType in the SDK headers (Enum/EBoardType.h), counted from 0.
MULTISCAN_PAR_UT = 29
RECEIVE_PARALLEL = 0x20   # Beam.receiveMode of a parallel multichannel beam

PACKET = 1024
PACKET_HEADER = 24
POINTS_PER_PACKET = (PACKET - PACKET_HEADER) // 2   # 500
DATA_16BIT, DATA_VERSION = 0x5A, 0x5B

VOLTAGES = (100, 200, 400)
# Receiver band-pass of a UT channel, MHz, indexed by Beam.filter
# (GroupEnum.h of the demo). 0 is all-pass, the board's full 0.5-25 MHz.
UT_FILTERS = (
    (0.5, 25.0), (0.9, 2.5), (1.5, 4.8), (2.8, 9.6),
    (4.5, 13.5), (5.5, 19.0), (9.5, 19.0),
)
# Beam.signalMode: 0 full wave, 1 positive half, 2 negative half, 3 RF.
RECTIFICATIONS = {"bipolar": 0, "unipolar_positive": 1,
                  "unipolar_negative": 2, "none": 3}
# Connector -> (channel index 0-7, pulse-echo). "CHn" sends and receives on
# the same socket; "CHn-PC" sends on T and listens on R of the same channel.
CONNECTORS = {}
for _ch in range(8):
    CONNECTORS[f"CH{_ch + 1}"] = (_ch, True)
    CONNECTORS[f"CH{_ch + 1}-PC"] = (_ch, False)



def sdk_channel(socket):
    """SDK channel index (0-7) of the socket labelled `socket + 1` on the case.

    The case and the firmware count from opposite ends. Measured: a 5 MHz
    probe on T1 rings only when SDK channel 8 fires, and SDK channel 1 alone
    shows nothing but the bare transmit pulse. Everything outside this module
    counts as the case does; only the SDK structs and packet headers use this.
    """
    return 7 - socket


socket_of = sdk_channel   # the mapping is its own inverse


DIGITIZER_MHZ = 100.0   # the board samples every 10 ns
MAX_SAMPLES = 32767     # kMaxUtPointNumber; a UT A-scan may not be longer
MAX_GAIN_DB = 110.0     # analog gain of a UT channel
# Manual, table 2.1: 25-1000 ns. Outside it the pulser cannot follow.
MIN_PULSE_NS, MAX_PULSE_NS = 25, 1000
# From the demo's Board.Prf (GetParallelUtMinWorkTime): a group needs its
# window plus this much dead time, and never less than MIN_BEAM_NS in total.
MIN_WAIT_NS, MIN_BEAM_NS, TX_DELAY_NS = 15000, 25000, 25000
MAX_PRF = 4000          # per channel, parallel board (LimitMultiScan400VPrf)


def pulse_width_ns(frequency_mhz):
    """Half a period of the probe frequency: 4 MHz -> 125 ns.

    The pulser fires one negative square pulse; a piezo is driven hardest when
    the pulse lasts half its resonance period.
    """
    if not frequency_mhz or frequency_mhz <= 0:
        raise ValueError("pulser.frequency must be greater than 0 MHz")
    width = int(round(500.0 / frequency_mhz))
    if not MIN_PULSE_NS <= width <= MAX_PULSE_NS:
        raise ValueError(
            f"{frequency_mhz} MHz needs a {width} ns pulse, but the pulser "
            f"only accepts {MIN_PULSE_NS}-{MAX_PULSE_NS} ns "
            f"({500.0 / MAX_PULSE_NS:.2f}-{500.0 / MIN_PULSE_NS:.2f} MHz)")
    return width


def range_ns(range_mm, velocity_m_s):
    """Pulse-echo time of flight across `range_mm`, in ns."""
    return int(round(2 * range_mm / velocity_m_s * 1e6))


def depth_to_samples(range_mm, velocity_m_s, compression):
    """A-scan sample count covering `range_mm` of pulse-echo depth."""
    samples = int(round(range_ns(range_mm, velocity_m_s) / 1000.0
                        * DIGITIZER_MHZ / compression))
    if samples > MAX_SAMPLES:
        raise ValueError(
            f"{range_mm} mm at {velocity_m_s} m/s needs {samples} samples "
            f"(max {MAX_SAMPLES}); reduce material.range or raise compression")
    return samples


def max_prf(group_ns, voltage=100, width_ns=0):
    """Highest frame PRF for groups that take `group_ns` in total per frame."""
    by_time = int(1e9 // group_ns)
    limit = MAX_PRF
    if voltage > 200:
        # A 400 V pulser needs time to recharge, more for a wider pulse.
        limit = 4000 if width_ns < 200 else 1000 if width_ns <= 250 else 500 if width_ns <= 500 else 250
    return min(by_time, limit)


def beam_time_ns(start_ns, window_ns, points, channels=1):
    """How long one group occupies the board, before the PRF stretches it.
    The upload of every channel's samples (20 ns a point) is part of it."""
    return max(start_ns + window_ns + MIN_WAIT_NS + points * 20 * channels
               + TX_DELAY_NS, MIN_BEAM_NS)


# --- SDK structures, byte for byte as in the headers (#pragma pack 1) -------

def _struct(name, fields):
    return type(name, (C.Structure,), {"_pack_": 1, "_fields_": fields})


Encoder = _struct("Encoder", [("encoderMode", C.c_int), ("polarity", C.c_int),
                              ("resolution", C.c_float), ("stop", C.c_bool)])
PaTofd = _struct("PaTofd", [f for i in range(1, 9) for f in (
    (f"ut{i}EmissionDamping", C.c_int), (f"ut{i}Pr", C.c_bool),
    (f"ut{i}ReceivingDamping", C.c_int))])
Global = _struct("Global", [
    ("beamNum", C.c_int), ("encoder1", Encoder), ("encoder2", Encoder),
    ("encoder3", Encoder), ("paTofd", PaTofd), ("paVoltage", C.c_int),
    ("paPolarityVoltage", C.c_int), ("stopUdp", C.c_bool),
    ("triggerMode", C.c_int), ("utVoltage", C.c_int), ("waveHeight", C.c_int),
    ("isFmcMode", C.c_bool)])
FpgaTfm = _struct("FpgaTfm", [(n, C.c_int) for n in (
    "aperture", "batchIndex", "captureRange", "captureStart", "columnNum",
    "depthNum", "fpgaTfmGroupIndex", "imageMode", "pieceNum",
    "receiveElementNum", "rowNum", "softGain", "soundPathMode")]
    + [("transmitElementId", C.c_int * 8), ("transmitInterval", C.c_int),
       ("angleNum", C.c_int)])
PaUt = _struct("PaUt", [(f"ut{i}Open", C.c_bool) for i in range(1, 9)])
MultiTransmit = _struct("MultiTransmit", [(n, C.c_uint) for n in (
    "txPulseNum", "txPulseSpace") + tuple(
    f"wave{i}{s}Voltage" for i in range(1, 5) for s in ("Left", "Right"))])
ParallelUt = _struct("ParallelUt", [
    ("groupId", C.c_int), ("utId", C.c_int), ("delay", C.c_uint),
    ("transmitSwitch", C.c_bool), ("receiveSwitch", C.c_bool)])
Tfm = _struct("Tfm", [("receiveElement", C.c_int), ("sampleMode", C.c_int),
                      ("samplePointNum", C.c_int)])
DynamicDepthFocus = _struct("DynamicDepthFocus", [
    ("onOff", C.c_bool), ("focusNumber", C.c_uint),
    ("receiveDelay", C.POINTER(C.POINTER(C.c_uint))),
    ("validDistance", C.POINTER(C.c_uint)), ("validDistanceLength", C.c_int),
    ("receiveUseElement", C.POINTER(C.c_int)),
    ("receiveUseElementLength", C.c_int), ("transmitReceiveDelay", C.c_int),
    ("isAnyDdfOn", C.c_bool)])
Beam = _struct("Beam", [
    ("average", C.c_int), ("beamId", C.c_int), ("envelope", C.c_bool),
    ("tfmFilterSwitch", C.c_bool), ("filter", C.c_int), ("fpgaTfm", FpgaTfm),
    ("gain", C.c_float), ("nextBeamId", C.c_int), ("paUt", PaUt),
    ("pointNum", C.c_int), ("parallelNum", C.c_int), ("pulseWidth", C.c_float),
    ("rangeNs", C.c_int), ("receiveElement", C.c_int),
    ("receiveMode", C.c_int), ("signalMode", C.c_int), ("startNs", C.c_int),
    ("tfm", Tfm), ("tgcNum", C.c_int), ("useTgc", C.c_bool),
    ("workTime", C.c_int), ("gainChangeTime", C.c_int),
    ("digitalGain", C.c_float), ("useParallel", C.c_bool),
    ("parallelBeamId", C.c_int), ("parallelTransmitBeamId", C.c_int),
    ("multiTransmit", MultiTransmit), ("parallelUt", ParallelUt),
    ("dynamicDepthFocus", DynamicDepthFocus), ("filterIndex", C.c_int)])
Tgc = _struct("Tgc", [
    ("beamId", C.c_int), ("dGainsLength", C.c_int),
    ("dGains", C.POINTER(C.c_float)), ("positionsLength", C.c_int),
    ("positions", C.POINTER(C.c_uint)), ("sampleRangeNs", C.c_int),
    ("sampleStartNs", C.c_int), ("tgcPointNum", C.c_int),
    ("wedgeDelay", C.c_int), ("parallelUt", ParallelUt)])
ParallelUtGroup = _struct("ParallelUtGroup", [("beamArr", Beam * 8),
                                              ("tgcArr", Tgc * 8)])


def flow_geometry(cfg):
    """(volumetric flow in l/s, liquid velocity in m/s) from the `flow` section.

    The sand rate is a mass per second; dividing it by the liquid passing in
    the same second is what turns it into a concentration.
    """
    flow = cfg.get("flow") or {}
    if "rate" not in flow or "diameter" not in flow:
        raise ValueError("config.yaml needs a flow section with rate (m3/h) "
                         "and diameter (mm)")
    if flow["rate"] <= 0 or flow["diameter"] <= 0:
        raise ValueError("flow.rate and flow.diameter must be greater than 0")
    litres_s = float(flow["rate"]) / 3.6                       # m3/h -> l/s
    area = math.pi * (float(flow["diameter"]) / 1000.0) ** 2 / 4.0   # m2
    return litres_s, litres_s / 1000.0 / area


HERE = os.path.dirname(os.path.abspath(__file__))


def beside(path):
    """A path as given, or the same name next to the program.

    The program is started from a shortcut, from an editor's Run button, from a
    terminal in some other directory - and a relative `config.yaml` means the
    current directory, which is usually not the one the program lives in. The
    result was a FileNotFoundError before anything opened, which reads as "it
    will not start without the instrument" and has nothing to do with the
    instrument. Anything not found where it was asked for is looked for beside
    the program instead.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(HERE, path)


def replace_file(tmp, path, tries=8):
    """`os.replace`, which on Windows sometimes needs asking twice.

    A virus scanner or the search indexer opens a file the moment it appears,
    and the replace then fails with "Access is denied" for the few milliseconds
    it holds it. Observed on this machine while writing calibration points, and
    a lost point is a lost injection - so the write waits and asks again rather
    than giving up on the first refusal.
    """
    for attempt in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _yaml_value(value):
    """One scalar as YAML writes it. Empty is how this file spells "not set"."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    return text if re.fullmatch(r"[A-Za-z0-9_./+-]+", text) else f'"{text}"'


# `  key: value  # comment` - the three parts that have to survive a rewrite.
_SETTING = re.compile(r"^(?P<indent>\s+)(?P<key>[A-Za-z_][\w]*):"
                      r"(?P<value>[^#]*)(?P<comment>#.*)?$")
_SECTION = re.compile(r"^(?P<key>[A-Za-z_][\w]*):")


def save_config(path, cfg, keys):
    """Write `section.key` values back into the config file, in place.

    A `yaml.safe_dump` would be one line and would erase every comment in it -
    and in this file the comments are the documentation of the parameters, the
    voltages that are accepted, what a band-pass has to be one of, which value
    breaks what. So each value is replaced where it stands, keeping the
    indentation, the ordering and whatever note sits at the end of the line.
    Keys the file does not have yet are appended to their section.

    Returns the keys actually written.
    """
    path = beside(path)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    wanted = {}
    for key in keys:
        section, name = key.split(".")
        wanted[(section, name)] = (cfg.get(section) or {}).get(name)

    written, section = set(), None
    ends = {}                       # last line of each section, for appending
    for i, line in enumerate(lines):
        head = _SECTION.match(line)
        if head:
            section = head.group("key")
        if section is not None and line.strip() and not line.lstrip().startswith("#"):
            ends[section] = i
        setting = _SETTING.match(line)
        if not setting or section is None:
            continue
        target = (section, setting.group("key"))
        if target not in wanted or target in written:
            continue
        value = _yaml_value(wanted[target])
        comment = setting.group("comment")
        rebuilt = f"{setting.group('indent')}{setting.group('key')}:"
        if value:
            rebuilt += f" {value}"
        if comment:
            # Keep the comment where it was, so a column of them stays a column.
            pad = max(1, len(line) - len(line.rstrip()) + len(setting.group("value")) - len(value) - 1)
            rebuilt += " " * pad + comment
        lines[i] = rebuilt.rstrip()
        written.add(target)

    for target in sorted(set(wanted) - written):
        section, name = target
        line = f"  {name}:"
        value = _yaml_value(wanted[target])
        if value:
            line += f" {value}"
        if section in ends:
            lines.insert(ends[section] + 1, line)
            ends = {s: (i + 1 if i > ends[section] else i) for s, i in ends.items()}
            ends[section] += 1
        else:
            lines += [f"{section}:", line]
            ends[section] = len(lines) - 1
        written.add(target)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    replace_file(tmp, path)         # never leave a half-written config
    return sorted(f"{s}.{n}" for s, n in written)


def load_config(path="config.yaml"):
    with open(beside(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def connectors_of(cfg):
    """`acquisition.connector` as a list: one name, or several for several beams."""
    names = cfg["acquisition"]["connector"]
    names = [names] if isinstance(names, str) else list(names)
    for name in names:
        if name not in CONNECTORS:
            raise ValueError(f"acquisition.connector {name!r} must be one of "
                             f"{list(CONNECTORS)}")
    if len({CONNECTORS[n][0] for n in names}) != len(names):
        raise ValueError("acquisition.connector uses the same channel twice")
    return names


# One channel as the board sees it. `transmit=False` is a passive channel: the
# receiver opens its window on every frame, the pulser stays silent.
# `sync=True` puts it in the same group as the channel before it, so both open
# their windows at the same instant: a passive channel synced to an active one
# hears that channel's pulse (through-transmission, or a test stimulus).
ChannelSpec = namedtuple("ChannelSpec", "channel transmit pulse_echo width_ns gain "
                         "band rectification start_ns window_ns points sync",
                         defaults=(False,))


def check_spec(c):
    name = f"CH{c.channel + 1}"
    if not 0 <= c.channel <= 7:
        raise ValueError("channel must be 1-8")
    if c.rectification not in RECTIFICATIONS:
        raise ValueError(f"{name}: rectification must be one of {list(RECTIFICATIONS)}")
    if tuple(c.band) not in UT_FILTERS:
        bands = ", ".join(f"{lo}-{hi}" for lo, hi in UT_FILTERS)
        raise ValueError(f"{name}: filter {c.band[0]}-{c.band[1]} MHz is not "
                         f"supported; pick one of: {bands}")
    if not 0 <= c.gain <= MAX_GAIN_DB:
        raise ValueError(f"{name}: gain must be 0-{MAX_GAIN_DB} dB")
    if c.transmit and not MIN_PULSE_NS <= c.width_ns <= MAX_PULSE_NS:
        raise ValueError(f"{name}: pulse must be {MIN_PULSE_NS}-{MAX_PULSE_NS} ns")
    if c.start_ns < 0 or c.window_ns <= 0:
        raise ValueError(f"{name}: start must be >= 0 and the window > 0")
    top = min(MAX_SAMPLES, c.window_ns // 10)
    if not 32 <= c.points <= top:
        raise ValueError(f"{name}: {c.points} points; the board takes 32 to "
                         f"{top} for a {c.window_ns / 1000:g} us window")


def grouped(specs):
    """Specs split into firing groups: a synced spec joins the one before it."""
    groups = []
    for c in specs:
        if c.sync and groups:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def group_ns(group):
    """Board time one group needs: its longest member, with every member's
    upload counted (they share the link)."""
    return max(beam_time_ns(c.start_ns, c.window_ns, c.points, len(group))
               for c in group)


def frame_ns(specs):
    """Board time one frame of these channels needs, at the fastest PRF."""
    return sum(group_ns(g) for g in grouped(specs))


def rate_limit(voltage, specs):
    """Highest frame PRF these channels can run at."""
    widths = [c.width_ns for c in specs if c.transmit] or [0]
    return max_prf(frame_ns(specs), voltage, max(widths))


def build_groups(voltage, rate, specs):
    """What the SDK is sent for these channels: (Global, [group]).

    Groups fire one after another in the order given; channels inside a group
    (see `sync`) fire together. Everything is validated before anything
    reaches the board: the SDK calls return nothing, so a bad value would only
    show as no data.
    """
    if voltage not in VOLTAGES:
        raise ValueError(f"voltage must be one of {VOLTAGES}")
    if not specs:
        raise ValueError("no channel is switched on")
    for c in specs:
        check_spec(c)
    if len({c.channel for c in specs}) != len(specs):
        raise ValueError("the same channel is used twice")
    if rate <= 0:
        raise ValueError("rate must be greater than 0 Hz")
    limit = rate_limit(voltage, specs)
    if rate > limit:
        # The board would silently run slower; refuse instead.
        raise ValueError(f"rate {rate:g} Hz is too fast for these channels: "
                         f"max {limit} Hz")
    firing = grouped(specs)
    # The frame period is shared out in proportion to what each group needs,
    # as the vendor software does; the board fires on these work times.
    need = [group_ns(g) for g in firing]
    stretch = 1e9 / rate / sum(need)

    glob = Global(beamNum=len(firing), utVoltage=voltage,
                  paVoltage=100, triggerMode=0, stopUdp=False)
    for enc in (glob.encoder1, glob.encoder2, glob.encoder3):
        enc.encoderMode, enc.resolution, enc.stop = 2, 67.0, True

    groups = []
    for group_id, members in enumerate(firing):
        work = int(math.ceil(need[group_id] * stretch))
        by_slot = {sdk_channel(c.channel): c for c in members}
        for ut, c in by_slot.items():
            n = ut + 1
            # 200 ohm damping above 300 V with a wide pulse, as the vendor does.
            damp = 1 if c.transmit and voltage >= 300 and c.width_ns >= 200 else 0
            setattr(glob.paTofd, f"ut{n}Pr", c.pulse_echo)
            setattr(glob.paTofd, f"ut{n}EmissionDamping", damp)
            setattr(glob.paTofd, f"ut{n}ReceivingDamping", damp)

        group = ParallelUtGroup()
        group._keep = []
        for ut in range(8):
            # All eight slots go out; only this group's channels are switched on.
            c = by_slot.get(ut)
            switch = ParallelUt(groupId=group_id, utId=ut, delay=0,
                                transmitSwitch=bool(c and c.transmit),
                                receiveSwitch=c is not None)
            b = group.beamArr[ut]
            b.parallelUt, b.average, b.workTime = switch, 1, work
            b.dynamicDepthFocus.focusNumber = 1
            start, window = (c.start_ns, c.window_ns) if c else (0, 1000)
            if c:
                b.beamId, b.nextBeamId = group_id, (group_id + 1) % len(firing)
                b.filter = UT_FILTERS.index(tuple(c.band))
                b.gain = float(c.gain)
                b.pointNum, b.startNs, b.rangeNs = c.points, start, window
                # The vendor software sends the width 15 ns short: that is
                # what the pulser adds on its own.
                b.pulseWidth = float(max(c.width_ns - 15, 15))
                b.receiveElement, b.receiveMode = 1, RECEIVE_PARALLEL
                b.signalMode = RECTIFICATIONS[c.rectification]
            # TCG is off, but the SDK reads the point arrays regardless: two
            # flat points, 0 dB, kept alive on the group.
            t = group.tgcArr[ut]
            t.parallelUt, t.beamId = switch, group_id
            t.sampleStartNs, t.sampleRangeNs = start, window
            t.tgcPointNum = t.dGainsLength = t.positionsLength = 2
            arrays = ((C.c_float * 2)(0.0, 0.0), (C.c_uint * 2)(0, window))
            group._keep.append(arrays)
            t.dGains, t.positions = arrays
        groups.append(group)
    return glob, groups


def specs_of(cfg):
    """The sand program's config.yaml as channel specs: active, one per connector."""
    pulser, receiver = cfg["pulser"], cfg["receiver"]
    acq, material = cfg["acquisition"], cfg["material"]
    if acq["start"] < 0:
        raise ValueError("acquisition.start (ns) cannot be negative")
    if acq["compression"] < 1:
        raise ValueError("acquisition.compression must be 1 or more")
    width = pulse_width_ns(pulser["frequency"])
    window = range_ns(material["range"], material["velocity"])
    points = depth_to_samples(material["range"], material["velocity"],
                              acq["compression"])
    band = (receiver["filter_min"], receiver["filter_max"])
    return [ChannelSpec(CONNECTORS[name][0], True, CONNECTORS[name][1], width,
                        receiver["gain"], band, receiver["rectification"],
                        int(acq["start"]), window, points)
            for name in connectors_of(cfg)]


def build_board_config(cfg):
    """The sand program's config.yaml -> (Global, [group])."""
    return build_groups(cfg["pulser"]["voltage"], cfg["acquisition"]["rate"],
                        specs_of(cfg))


Frame = namedtuple("Frame", "channel cycle ascan encoder_a encoder_b")


def _load_sdk():
    """The vendor DLL, from the SDK folder beside this file or from PATH."""
    path = os.environ.get("DPLBOARD_DLL") or os.path.join(
        HERE, "CPluPlusSdk_v2026.06.17", "bin", "DplBoard.dll")
    archive = os.path.join(HERE, "sdk", "CPluPlusSdk_v2026.06.17.zip")
    if not os.path.exists(path) and not os.environ.get("DPLBOARD_DLL") and os.path.exists(archive):
        # A fresh clone carries the SDK zipped; the DLL is all that is needed.
        import zipfile
        with zipfile.ZipFile(archive) as z:
            z.extract("CPluPlusSdk_v2026.06.17/bin/DplBoard.dll", HERE)
    try:
        dll = C.CDLL(path if os.path.exists(path) else "DplBoard.dll")
    except OSError as e:
        raise OSError(f"could not find or load DplBoard.dll ({e}). Copy the "
                      "vendor's C++ SDK folder CPluPlusSdk_v2026.06.17 next to "
                      "multiscan.py, or set DPLBOARD_DLL to the x64 DLL") from None
    for name, args in {
        "RegisterC": [C.c_void_p], "SetLocalIP": [C.c_char_p],
        "AddBoard": [C.c_int] * 4, "RemoveAllBoard": [], "Connect": [],
        "DisConnect": [], "Dispose": [], "GetBoardVersion": [C.c_int],
        "StartWork": [], "StopWork": [],
        "SendGlobal": [C.c_int, C.POINTER(Global)],
        "SendParallelUtGroup": [C.c_int, C.POINTER(ParallelUtGroup)],
    }.items():
        fn = getattr(dll, name)
        fn.argtypes, fn.restype = args, None
    return dll


_CALLBACK = C.CFUNCTYPE(None, C.POINTER(C.c_ubyte), C.c_int)
_sdk = None        # one DLL per process, and with it one callback
_packets = queue.SimpleQueue()


@_CALLBACK
def _on_packet(data, length):
    # Runs on the SDK's receive thread: copy and leave, nothing else.
    _packets.put(C.string_at(data, length))


def _drain():
    while True:
        try:
            _packets.get_nowait()
        except queue.Empty:
            return


def local_ip_for(ip, port=DEFAULT_PORT):
    """The address of the network card that routes to `ip`. Sends nothing."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((ip, port))
        return s.getsockname()[0]


class MultiScan:
    """One parallel Multiscan board.

        dev = MultiScan("192.168.1.130")
        dev.configure(cfg)            # the pulsers start firing
        frame = dev.recv()            # Frame(channel, cycle, ascan, ...)
        dev.close()                   # and stop

    `recv_frame()` is the same call in the OmniscanPy client's shape,
    (64-byte header, int16 A-scan), which is what the web app reads.
    """

    def __init__(self, ip=DEFAULT_IP, port=DEFAULT_PORT, local_ip=None):
        global _sdk
        if _sdk is None:
            _sdk = _load_sdk()
            _sdk.RegisterC(C.cast(_on_packet, C.c_void_p))
        self.ip, self.port = ip, port
        self.local_ip = local_ip or local_ip_for(ip, port)
        self.points = {}           # channel (0-7) -> A-scan length
        self._parts = {}           # channel -> (frame id, {package: samples})
        self._cycle = {}           # channel -> (last frame id, unwrapped count)
        self.version = None
        self._glob = None
        _drain()                   # nothing from an earlier board may leak in
        # Several network cards (Wi-Fi, VPN, WSL) are the rule on a laptop;
        # the SDK must listen on the one that reaches the board.
        # The SDK listens on the board's port number locally. If another
        # program (an old scope.py, the vendor software) holds it, the SDK
        # binds nothing and every call silently goes unanswered - so ask first.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # Exclusive: the SDK shares its socket (SO_REUSEADDR), so a plain
            # bind next to it succeeds and two programs end up both driving
            # the board - observed: two scope.py at once, each half the data.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind((self.local_ip, port))
            except OSError:
                raise RuntimeError(f"UDP port {port} on {self.local_ip} is held by "
                                   "another program (an old scope.py/bscan_web.py, "
                                   "or the vendor software); close it first") from None
        _sdk.SetLocalIP(self.local_ip.encode())
        _sdk.RemoveAllBoard()
        _sdk.AddBoard(0, MULTISCAN_PAR_UT, int(ip.rsplit(".", 1)[1]), port)
        _sdk.Connect()             # UDP: nothing is checked, it just binds

    def configure(self, cfg):
        """Set up from the sand program's config.yaml."""
        self.configure_channels(cfg["pulser"]["voltage"],
                                cfg["acquisition"]["rate"], specs_of(cfg))

    def configure_channels(self, voltage, rate, specs):
        """Set up any mix of active and passive channels (ChannelSpec list)."""
        glob, groups = build_groups(voltage, rate, specs)
        self.points = {c.channel: c.points for c in specs}
        self._glob = glob
        _drain()     # A-scans of the previous setup must not count as this one's
        self._parts.clear()
        self._cycle.clear()
        _sdk.SendGlobal(0, C.byref(glob))
        for group in groups:
            _sdk.SendParallelUtGroup(0, C.byref(group))

    def get_version(self, timeout=2.0):
        """(PA FPGA, UT FPGA) version words. Also the cheapest "is it there"."""
        self.version = None
        _sdk.GetBoardVersion(0)
        end = time.monotonic() + timeout
        while self.version is None and time.monotonic() < end:
            self._next_packet(end - time.monotonic())
        if self.version is None:
            raise TimeoutError(f"no answer from {self.ip}:{self.port} "
                               f"(local address {self.local_ip})")
        return self.version

    def _next_packet(self, timeout):
        """Handle one packet; return a Frame if it completed an A-scan."""
        try:
            pkt = _packets.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return None
        kind = pkt[0]
        if kind == DATA_VERSION:
            self.version = (int.from_bytes(pkt[2:6], "big"),
                            int.from_bytes(pkt[6:10], "big"))
            return None
        if kind != DATA_16BIT or not self.points:
            return None
        channel = socket_of(pkt[22] - 1)   # header: SDK channel 1-8
        n = self.points.get(channel)
        if n is None:
            return None
        frame_id, package = pkt[2], (pkt[7] << 8) | pkt[8]
        count = math.ceil(n / POINTS_PER_PACKET)
        if package >= count:
            return None              # left over from a longer A-scan
        start = package * POINTS_PER_PACKET
        take = min(POINTS_PER_PACKET, n - start)
        samples = np.frombuffer(pkt, ">i2", take, PACKET_HEADER)

        fid, parts = self._parts.get(channel, (None, None))
        if fid != frame_id:
            parts = {}               # a new A-scan: a half-built one is lost
            self._parts[channel] = (frame_id, parts)
        parts[package] = samples
        if len(parts) < count:
            return None
        del self._parts[channel]
        ascan = np.concatenate([parts[i] for i in range(count)]).astype(np.int16)

        # The frame id counts A-scans modulo 256; unwrapped, it is the cycle
        # counter SequenceTracker uses to count losses and time each A-scan.
        last, cycle = self._cycle.get(channel, (None, 0))
        if last is not None:
            cycle += (frame_id - last) % 256 or 256
        self._cycle[channel] = (frame_id, cycle)
        return Frame(channel, cycle, ascan,
                     int.from_bytes(pkt[9:13], "big", signed=True),
                     int.from_bytes(pkt[13:17], "big", signed=True))

    def recv(self, timeout=3.0):
        """The next complete A-scan of any channel. TimeoutError if none comes."""
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no A-scan from {self.ip} for {timeout} s")
            frame = self._next_packet(left)
            if frame is not None:
                return frame

    def recv_frame(self, timeout=3.0):
        """(header, A-scan) as the OmniscanPy client returns them: the header is
        64 bytes of uint32 whose word 0 is the cycle counter, word 1 the channel."""
        f = self.recv(timeout)
        header = np.zeros(16, np.uint32)
        header[0], header[1] = f.cycle & 0xFFFFFFFF, f.channel
        return header.tobytes(), f.ascan

    def close(self):
        try:
            if self._glob is not None:
                # Stop the upload; the board has no other "stop".
                self._glob.stopUdp = True
                _sdk.SendGlobal(0, C.byref(self._glob))
        finally:
            _sdk.DisConnect()
            _sdk.RemoveAllBoard()


class SequenceTracker:
    """Counts A-scans lost between the instrument and this program.

    The 64-byte header layout is undocumented, so the cycle counter is located
    at run time: it is the uint32 field that grows by exactly one per frame.
    Once found, any larger step means the publisher dropped frames.
    """

    def __init__(self):
        self.offset = None    # header word holding the counter
        self.dropped = 0
        self.cycle = None     # the counter's value on the last frame
        self.step = 1         # cycles since the frame before it
        self._prev = None
        self._candidates = None

    def update(self, header):
        """Follow the counter. `step` is how many acquisition cycles have
        passed since the last frame, which is the only honest measure of how
        long an A-scan stands for: the instrument fires on its own timer, but
        the frames reach this program in bursts of several at a time with tens
        of milliseconds between the bursts. Timing an A-scan by when its packet
        arrived credits one member of a burst with the whole gap and the rest
        with nothing.
        """
        words = np.frombuffer(header, dtype=np.uint32)
        self.step = 1
        if self.offset is None:
            if self._prev is not None:
                step = words.astype(np.int64) - self._prev.astype(np.int64)
                self._candidates = {i for i in self._candidates if step[i] == 1}
                if len(self._candidates) == 1:
                    self.offset = next(iter(self._candidates))
                    self.cycle = int(words[self.offset])
            else:
                self._candidates = set(range(len(words)))
            self._prev = words
            return
        step = int(words[self.offset]) - int(self._prev[self.offset])
        if step > 1:
            self.dropped += step - 1
        self.step = max(1, step)
        self.cycle = int(words[self.offset])
        self._prev = words


class Recording:
    """Appends raw A-scans to disk: a flat int16 .bin plus a .json sidecar.

    Raw means raw: full depth, before the crop, before anything is subtracted.
    What it takes to reproduce the processing goes in the sidecar instead, so a
    recording can be re-read under a different chain later.
    """

    def __init__(self, directory, cfg):
        directory = beside(directory)
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"ut_{stamp}.bin")
        self._file = open(self.path, "wb")
        self.count = 0
        self.samples = None
        # All three are set once the warm-up block has been reduced.
        self.baseline = None      # the reference A-scan, cropped
        self.svd_basis = None     # (k, n) or None when the stage is off
        self.vlim = None          # B-scan colour limit for the run
        self._cfg = cfg

    def write(self, ascan):
        if self.samples is None:
            self.samples = len(ascan)
        self._file.write(ascan.tobytes())
        self.count += 1

    def flush(self):
        """Make everything written so far visible to a separate reader."""
        self._file.flush()

    def close(self):
        self._file.close()
        meta = {
            "samples": self.samples,
            "ascans": self.count,
            "dtype": "int16",
            "rate_hz": self._cfg["acquisition"]["rate"],
            "range_mm": self._cfg["material"]["range"],
            "velocity_m_s": self._cfg["material"]["velocity"],
            "baseline_shots": self._cfg["baseline"]["shots"],
            "processing": self._cfg.get("processing") or {},
            "analysis": self._cfg.get("analysis") or {},
            "vlim": self.vlim,
            "baseline": (None if self.baseline is None
                         else [round(float(v), 2) for v in self.baseline]),
            # A basis row per line; k is small, so plain JSON stays readable.
            "svd_basis": (None if self.svd_basis is None
                          else [[round(float(v), 6) for v in row]
                                for row in self.svd_basis]),
        }
        with open(self.path.replace(".bin", ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def load_recording(bin_path, subtract_baseline=False):
    """Return (bscan, meta) where bscan has one raw A-scan per row.

    With `subtract_baseline`, rows are cropped to the reference's length and
    the reference is removed - the first two stages of the live chain. The
    band-pass and the SVD projection are not applied here; `pipeline.Pipeline`
    reproduces the whole chain from the same sidecar when that is wanted.
    """
    with open(bin_path.replace(".bin", ".json"), encoding="utf-8") as f:
        meta = json.load(f)
    bscan = np.fromfile(bin_path, dtype=np.int16).reshape(-1, meta["samples"])
    if subtract_baseline and meta.get("baseline"):
        baseline = np.array(meta["baseline"], dtype=np.float64)
        bscan = bscan[:, :len(baseline)].astype(np.float64) - baseline
    return bscan, meta


def _packet(channel, frame_id, package, samples):
    """One board packet, as the board sends it (for the self-check)."""
    head = bytearray(PACKET_HEADER)
    head[0], head[1], head[2] = DATA_16BIT, 130, frame_id
    head[7], head[8], head[22], head[23] = package >> 8, package & 0xFF, sdk_channel(channel) + 1, 0xA5
    body = np.asarray(samples, ">i2").tobytes()
    return bytes(head) + body + bytes(PACKET - PACKET_HEADER - len(body))


if __name__ == "__main__":
    # Struct sizes as g++ reports them for the SDK headers (#pragma pack 1).
    assert (C.sizeof(Global), C.sizeof(Beam), C.sizeof(Tgc),
            C.sizeof(ParallelUtGroup)) == (137, 292, 58, 2800)
    assert Beam.workTime.offset == 171 and Global.utVoltage.offset == 128
    assert pulse_width_ns(4.0) == 125
    assert depth_to_samples(60.0, 1500.0, 1) == 8000      # 80 us at 100 MHz
    q, v = flow_geometry({"flow": {"rate": 12.0, "diameter": 100.0}})
    assert abs(q - 3.3333) < 1e-3 and abs(v - 0.4244) < 1e-3, (q, v)

    # Reassembly, without a board: 1200 points = 3 packets, out of order,
    # then a frame id wrap 255 -> 0 that must count as one cycle.
    dev = MultiScan.__new__(MultiScan)
    dev.points, dev._parts, dev._cycle = {2: 1200}, {}, {}
    ramp = np.arange(-600, 600, dtype=np.int16)
    for fid in (255, 0):
        for pkg in (1, 0, 2):
            _packets.put(_packet(2, fid, pkg, ramp[pkg * 500:(pkg + 1) * 500]))
    _packets.put(_packet(5, 0, 0, ramp[:500]))            # a channel not asked for
    first, second = dev.recv(1.0), dev.recv(1.0)
    assert first.channel == 2 and np.array_equal(first.ascan, ramp)
    assert second.cycle - first.cycle == 1, (first.cycle, second.cycle)
    tracker = SequenceTracker()
    for f in (first, second):
        tracker.update(np.array([f.cycle] + [0] * 15, np.uint32).tobytes())
    assert tracker.offset == 0 and tracker.dropped == 0

    cfg = load_config()
    glob, groups = build_board_config(cfg)
    on = [b for g in groups for b in g.beamArr if b.parallelUt.transmitSwitch]
    print(f"config.yaml OK -> {on[0].pointNum} samples for "
          f"{cfg['material']['range']} mm at {cfg['material']['velocity']} m/s, "
          f"{len(groups)} channel(s), pulse {pulse_width_ns(cfg['pulser']['frequency'])} ns, "
          f"max rate {rate_limit(cfg['pulser']['voltage'], specs_of(cfg))} Hz")

    # A passive channel opens its receiver and keeps its pulser off.
    passive = ChannelSpec(2, False, True, 0, 40.0, (0.5, 25.0), "none", 0, 5_000_000, 25000)
    _, (group,) = build_groups(100, 50, [passive])
    switch = group.beamArr[sdk_channel(2)].parallelUt
    assert switch.receiveSwitch and not switch.transmitSwitch

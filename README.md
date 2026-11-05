# MultiFlow: live flow imaging

This is a new app. MultiView, MultiDAQ and MultiAnalysis are unchanged.

MultiFlow acquires A-scans from the probes at 12, 3, 6 and 9 o'clock and processes every one as it arrives. It shows what the MultiAnalysis2 videos showed, but live:

- **the cross-section now**;
- **the vertical (12–6) and horizontal (9–3) planes** scrolling in time;
- **gas, scatter near the bottom wall and activity**;
- **an event list**.

Recording works as in MultiView: you type a file name, Enter starts the next file, and Space adds a mark.

## Run

```
pip install -r requirements.txt
cd C:\Users\RavanKhidirov\Downloads\MultiAlert\MultiFlow
python flow_web.py              ->  open http://127.0.0.1:8012
python flow_web.py --demo       (simulated flow with sand bursts, no board needed)
```

The first time it runs, MultiFlow copies `channels.yaml` from `..\MultiView`, so your positions, pipe and calibration carry over. After that, Setup and Calibrate work exactly as in MultiView.

## Using it

1. **Start.** For the first **5 s** the app learns the *reference*: the normal flow for each probe. The pill shows "learning reference". Keep the flow normal during this time and don't inject anything.
2. **Watch:**
   - **Colour** is moving scatter: anything that changes from one A-scan to the next in the sand band (6–10 MHz), meaning sand and bubbles. It is shown in **dB relative to that probe's own far-wall echo**, so all four probes share one scale. The default scale runs from −50 to −25 dB and can be changed under the cross-section. Quiet flow sits at about −52 dB.
   - **Light blue** is gas, where no probe can see.
     - A probe stops counting behind a strong new echo.
     - Gas sitting on a probe's own wall blinds that probe completely.
     - A dashed line marks how far down the gas reaches.
   - **Tiles:**
     - gas fraction on the beam lines;
     - scatter near the bottom wall (it turns amber at or above the sand-like threshold);
     - activity;
     - the current event.
3. **Re-learn reference:** press it whenever the normal flow changes (new flow rate, temperature drift). The reference also follows slow drift on its own while nothing is happening (`adapt_tau_s`).
4. **Events:** an event starts when activity stays above `event_z` for 0.25 s and ends after 1.5 s of quiet. Each event is labelled:
   - **gas:** gas seen on at least 2 % of the beam lines;
   - **sand-like:** scatter near the bottom wall at or above `sand_db` (−46 dB);
   - **sand-like + gas:** both;
   - **activity:** neither.
5. **Replay** plays back any recorded session, including MultiView's (it lists `recordings` and `..\MultiView\recordings`). Use it to check settings on your earlier runs.

Calibrate (pipe full of water, no bubbles) still sets each probe's time zero. After calibrating, the reference is learned again.

## What is recorded

- **Raw A-scans:** the same files and the same `session.json` as MultiView, so MultiAnalysis, MultiAnalysis2 and `flow_export.py` work on these recordings unchanged.
- **`<file>_flow.csv`:** every 0.1 s, the activity, the gas fraction, the scatter near the bottom wall in dB, and how far down the gas reaches in mm.
- **`flow_events.csv`:** every event, with its file name, type, duration and maxima. An event still running when you stop, or move to the next file, is written as "(still going)".

## Settings (`config.yaml`, section `flow`)

| key | default | meaning |
|---|---|---|
| `learn_s` | 5 | seconds of normal flow used to learn the reference |
| `adapt_tau_s` | 120 | how fast the reference follows drift while nothing happens (0 = frozen) |
| `shadow_db` | 10 | a beam stops counting behind a new echo this strong |
| `gas_db` | 15 | behind an echo this strong the region counts as gas |
| `ring_db` | 3 | wall ringing up this much = gas on that probe's wall |
| `event_z` | 4 | activity threshold for events |
| `sand_db` | −46 | scatter near the bottom wall above this = sand-like |
| `gas_frac_min` | 0.02 | gas fraction above this = gas |

Where these numbers come from, on the 4 November runs:

| case | scatter near the bottom wall |
|---|---|
| sand | −41 to −37 dB |
| weakest sand run | −47.6 dB |
| air slugs | never above −49.5 dB |
| quiet flow (99th percentile) | about −52 dB |

These thresholds were set on only those 8 runs, so check them on new data. Both thresholds are drawn as dashed lines on the activity chart.

## Speed

Processing uses about 5 % of one CPU core for 4 probes at 120 Hz. Each band is mixed down to a 20 MHz complex signal straight from one FFT, and only the samples up to just past the far wall are used.

If the PC falls behind, recording never skips anything. The display skips instead, and the "not processed" count in the header goes up.

#!/usr/bin/env python3
"""
Analyse a Deep Trekker ROV log (CSV written by log_rov_csv.py): sampling,
dives, navigation data quality, sync manoeuvres, and overview plots.

    pip install numpy matplotlib

    python rov_analysis.py rov_20261113_173335.csv -o rov_report/
    python rov_analysis.py rov.csv --map heading=nav_heading --map depth=depth_m   # force column names

Columns are found by the end of their name (JSON path of the ROV state, e.g.
'velocity.forward', 'position.local.x', 'waterTemp'); the mapping used is
printed. Outputs (in -o):
  summary.json     sampling, dives, statistics, validity, sync manoeuvres
  overview.png     depth / altitude, attitude, DVL velocity, temperature / battery
  track.png        USBL local position and DVL dead reckoning

Conventions assumed for the Deep Trekker state: heading in degrees, clockwise
from north; pitch / roll in degrees; DVL velocity in m/s in the body frame
(forward, rightward, downward); depth and altitude in m.
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# key -> accepted column name endings (JSON paths of the Deep Trekker state, case-insensitive)
ALIASES = {
    "t": ["t_unix"],
    "depth": ["depth"],
    "altitude": ["altitude"],
    "altitude_valid": ["altitudeValid"],
    "heading": ["heading"],
    "pitch": ["pitch"],
    "roll": ["roll"],
    "vel_forward": ["velocity.forward"],
    "vel_right": ["velocity.rightward"],
    "vel_down": ["velocity.downward"],
    "vel_valid": ["velocity.valid"],
    "x": ["position.local.x"],
    "y": ["position.local.y"],
    "pos_uncertainty": ["position.local.uncertainty"],
    "water_temp": ["waterTemp"],
    "in_water": ["inWater"],
    "tilt": ["cameraHead.tilt.position", "tilt.position"],
    "battery_left": ["leftBattery.percent"],
    "battery_right": ["rightBattery.percent"],
}


def normalize(name):
    return re.sub(r"[._/ \-]+", ".", name.strip().lower())


def map_columns(columns, overrides=None):
    """{key: column} for the keys found in the CSV header."""
    overrides = overrides or {}
    mapping = {}
    norm = {c: normalize(c) for c in columns}
    for key, aliases in ALIASES.items():
        if key in overrides:
            if overrides[key] not in columns:
                sys.exit(f"--map {key}={overrides[key]}: column not in the CSV")
            mapping[key] = overrides[key]
            continue
        for alias in aliases:
            a = normalize(alias)
            hits = [c for c in columns if norm[c] == a or norm[c].endswith("." + a)]
            if hits:
                mapping[key] = min(hits, key=len)
                break
    return mapping


def parse_value(v):
    if v is None or v == "":
        return np.nan
    if v in ("True", "true"):
        return 1.0
    if v in ("False", "false"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return np.nan


def load_rov_csv(path, overrides=None):
    """Returns (data, mapping, raw): data[key] = float array for the mapped keys,
    raw[column] = float array for every numeric column; rows sorted by time."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    mapping = map_columns(columns, overrides)
    if "t" not in mapping:
        sys.exit(f"No time column (t_unix) in {path}; columns: {', '.join(columns)}")
    raw = {c: np.array([parse_value(r.get(c)) for r in rows]) for c in columns}
    raw = {c: v for c, v in raw.items() if np.isfinite(v).any()}
    order = np.argsort(raw[mapping["t"]], kind="stable")
    raw = {c: v[order] for c, v in raw.items()}
    t = raw[mapping["t"]]
    keep = np.isfinite(t) & np.concatenate([[True], np.diff(t) > 0])
    raw = {c: v[keep] for c, v in raw.items()}
    data = {k: raw[c] for k, c in mapping.items() if c in raw}
    return data, mapping, raw


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def segments(mask, t, min_duration=10.0, max_gap=5.0):
    """Contiguous [start, end] time intervals where mask is True."""
    out, start, last = [], None, None
    for ti, m in zip(t, mask):
        if m:
            if start is None or ti - last > max_gap:
                if start is not None and last - start >= min_duration:
                    out.append((start, last))
                start = ti
            last = ti
    if start is not None and last - start >= min_duration:
        out.append((start, last))
    return out


def stats(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), n=int(len(v)))


def dead_reckoning(t, heading, vf, vr, valid):
    """North / east track by integrating the DVL body velocity with the heading."""
    psi = np.radians(heading)
    dt = np.diff(t, prepend=t[0])
    ok = (valid > 0.5) & np.isfinite(vf) & np.isfinite(vr) & np.isfinite(psi) & (dt < 5.0)
    vn = np.where(ok, vf * np.cos(psi) - vr * np.sin(psi), 0.0)
    ve = np.where(ok, vf * np.sin(psi) + vr * np.cos(psi), 0.0)
    return np.cumsum(vn * dt), np.cumsum(ve * dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="ROV log written by log_rov_csv.py")
    ap.add_argument("-o", "--outdir", default="rov_report")
    ap.add_argument("--map", action="append", default=[], metavar="KEY=COLUMN",
                    help=f"force a column; keys: {', '.join(ALIASES)}")
    ap.add_argument("--yaw-rate", type=float, default=15.0,
                    help="yaw rate (deg/s) above which a turn counts as a sync manoeuvre")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    overrides = dict(m.split("=", 1) for m in a.map)
    data, mapping, raw = load_rov_csv(a.csv, overrides)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t = data["t"]
    rel = t - t[0]

    print("Column mapping:")
    for k in ALIASES:
        print(f"  {k:<16} {mapping.get(k, '-')}")
    other = [c for c in raw if c not in mapping.values()]
    print(f"  + {len(other)} other numeric columns (kept as raw topics in merge_bag.py)")

    # sampling
    dt = np.diff(t)
    med = float(np.median(dt)) if len(dt) else float("nan")
    gaps = [(float(t[i]), float(dt[i])) for i in np.where(dt > max(3 * med, 1.0))[0]]
    print(f"\nLog: {len(t)} rows, {iso(t[0])} -> {iso(t[-1])} ({(t[-1] - t[0]) / 60:.1f} min), "
          f"median period {med:.3f} s (~{1 / med:.1f} Hz), {len(gaps)} gaps")

    # dives
    if "in_water" in data:
        wet = data["in_water"] > 0.5
    elif "depth" in data:
        wet = data["depth"] > 0.3
    else:
        wet = np.ones(len(t), bool)
    dives = []
    for s, e in segments(wet, t):
        sel = (t >= s) & (t <= e)
        d = dict(start=iso(s), end=iso(e), start_unix=s, end_unix=e, duration_min=(e - s) / 60)
        if "depth" in data:
            d["max_depth_m"] = float(np.nanmax(data["depth"][sel]))
        dives.append(d)
    print(f"Dives: {len(dives)}")
    for i, d in enumerate(dives):
        print(f"  {i + 1}. {d['start']} -> {d['end']}  {d['duration_min']:.1f} min"
              + (f", max depth {d['max_depth_m']:.2f} m" if "max_depth_m" in d else ""))

    # statistics and validity
    summary_stats = {k: stats(v) for k, v in data.items() if k != "t"}
    validity = {}
    for key, flag in (("altitude", "altitude_valid"), ("dvl_velocity", "vel_valid")):
        if flag in data:
            validity[key] = float(100 * np.nanmean(data[flag][wet] > 0.5)) if wet.any() else None
    if "pos_uncertainty" in data:
        validity["usbl_uncertainty_median"] = float(np.nanmedian(data["pos_uncertainty"][wet])) if wet.any() else None
    print("Validity in water: " + ", ".join(
        f"{k} {v:.0f} %" if "uncertainty" not in k else f"{k} {v:.1f}" for k, v in validity.items() if v is not None))

    # sync manoeuvres: fast yaw turns
    manoeuvres = []
    if "heading" in data:
        h = np.degrees(np.unwrap(np.radians(data["heading"])))
        rate = np.gradient(h, t)
        for s, e in segments(np.abs(rate) > a.yaw_rate, t, min_duration=0.0, max_gap=2 * med + 0.5):
            sel = (t >= s) & (t <= e)
            manoeuvres.append(dict(time=iso(s), t_unix=s, duration_s=e - s,
                                   peak_yaw_rate_dps=float(np.max(np.abs(rate[sel])))))
        print(f"Yaw turns faster than {a.yaw_rate:g} deg/s (usable for synchronization): {len(manoeuvres)}")

    summary = dict(file=str(a.csv), mapping=mapping, rows=len(t), start=iso(t[0]), end=iso(t[-1]),
                   start_unix=float(t[0]), end_unix=float(t[-1]), median_period_s=med,
                   gaps=[dict(t_unix=g[0], duration_s=g[1]) for g in gaps], dives=dives,
                   stats=summary_stats, validity_percent=validity, sync_manoeuvres=manoeuvres)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out / 'summary.json'}")

    # figures
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [p for p in (
        ("depth / altitude (m)", [("depth", "depth"), ("altitude", "altitude")], True),
        ("attitude (deg)", [("pitch", "pitch"), ("roll", "roll"), ("tilt", "camera tilt")], False),
        ("heading (deg)", [("heading", "heading")], False),
        ("DVL velocity (m/s)", [("vel_forward", "forward"), ("vel_right", "rightward"), ("vel_down", "downward")], False),
        ("temperature (°C) / battery (%)", [("water_temp", "water"), ("battery_left", "battery L"),
                                            ("battery_right", "battery R")], False),
    ) if any(k in data for k, _ in p[1])]
    fig, ax = plt.subplots(len(panels), 1, figsize=(12, 2.3 * len(panels)), sharex=True, squeeze=False)
    for axis, (label, series, invert) in zip(ax[:, 0], panels):
        for key, name in series:
            if key in data:
                v = data[key].copy()
                if key == "altitude" and "altitude_valid" in data:
                    v[data["altitude_valid"] < 0.5] = np.nan
                if key.startswith("vel_") and "vel_valid" in data:
                    v[data["vel_valid"] < 0.5] = np.nan
                x = rel
                if key == "heading":  # break the line at the 360 -> 0 wrap
                    cut = np.where(np.abs(np.diff(v)) > 180)[0] + 1
                    x, v = np.insert(rel, cut, np.nan), np.insert(v, cut, np.nan)
                axis.plot(x, v, lw=0.8, label=name)
        if invert:
            axis.invert_yaxis()
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
        for d in dives:
            axis.axvspan(d["start_unix"] - t[0], d["end_unix"] - t[0], color="tab:blue", alpha=0.05)
        for m in manoeuvres:
            axis.axvline(m["t_unix"] - t[0], color="tab:red", lw=0.5, alpha=0.5)
    ax[-1, 0].set_xlabel(f"time since {iso(t[0])} (s)   red lines = fast yaw turns")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150)

    has_usbl = "x" in data and "y" in data
    has_dvl = all(k in data for k in ("heading", "vel_forward", "vel_right"))
    if has_usbl or has_dvl:
        fig, ax = plt.subplots(1, 2 if has_usbl and has_dvl else 1, figsize=(12 if has_usbl and has_dvl else 6, 5.5),
                               squeeze=False)
        k = 0
        if has_usbl:
            sc = ax[0, k].scatter(data["x"], data["y"], c=rel, s=4, cmap="viridis")
            ax[0, k].set_title("USBL local position")
            ax[0, k].set_xlabel("x")
            ax[0, k].set_ylabel("y")
            fig.colorbar(sc, ax=ax[0, k], label="time (s)")
            k += 1
        if has_dvl:
            valid = data.get("vel_valid", np.ones(len(t)))
            north, east = dead_reckoning(t, data["heading"], data["vel_forward"], data["vel_right"], valid)
            sc = ax[0, k].scatter(east, north, c=rel, s=4, cmap="viridis")
            ax[0, k].set_title("DVL dead reckoning (from the start of the log)")
            ax[0, k].set_xlabel("east (m)")
            ax[0, k].set_ylabel("north (m)")
            fig.colorbar(sc, ax=ax[0, k], label="time (s)")
        for axis in ax[0]:
            axis.axis("equal")
            axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "track.png", dpi=150)
    print(f"-> figures in {out}/")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()

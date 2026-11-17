#!/usr/bin/env python3
"""
Extract the IMU (raw + normalized) and the attitude (quaternions) from a
DJI Osmo Action 5 Pro MP4, using telemetry-parser (the Gyroflow library).

    pip install telemetry-parser
    python dji_imu_extract.py DJI_20261116_0001.MP4 [-o output/] [--dump-json]

Reminder: the camera only records these data with stabilization (EIS /
RockSteady) off and the FOV set to Wide.

Outputs (in -o, next to the MP4 by default):
  <name>_gyro_raw.csv, <name>_accl_raw.csv   sensor data as decoded
  <name>_imu_norm.csv                        IMU in the Gyroflow camera frame (gyro in deg/s)
  <name>_attitude.csv                        quaternions (w,x,y,z) + ZYX Euler angles in degrees
  <name>_telemetry.json                      (--dump-json) all the decoded content, for inspection

On the Action 5 Pro, telemetry-parser only returns the attitude quaternions;
the per-frame accelerometer is read by djmd_inspect.py / mp4_to_rosbag.py.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    import telemetry_parser
except ImportError:
    sys.exit("telemetry-parser missing: pip install telemetry-parser")


def walk(obj, path=()):
    """Walks a nested structure and yields (path, dict) for every dict."""
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk(v, path)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def collect(telemetry):
    """Finds, in the output of tp.telemetry(), the gyroscope / accelerometer /
    magnetometer series ({t,x,y,z} dicts) and the quaternions ({t, v:{w,x,y,z}}
    or {t,w,x,y,z}). Matches on shape rather than exact keys, to cope with
    telemetry-parser versions and camera models."""
    series = {"Gyroscope": [], "Accelerometer": [], "Magnetometer": []}
    quats = []
    for path, d in walk(telemetry):
        p = " ".join(path).lower()
        keys = set(d)
        if {"t", "x", "y", "z"} <= keys and "w" not in keys and all(is_num(d[k]) for k in "txyz"):
            for name in series:
                if name.lower() in p or name[:4].lower() in p:
                    series[name].append((d["t"], d["x"], d["y"], d["z"]))
                    break
        elif "t" in keys and isinstance(d.get("v"), dict) and {"w", "x", "y", "z"} <= set(d["v"]):
            v = d["v"]
            quats.append((d["t"], v["w"], v["x"], v["y"], v["z"]))
        elif {"t", "w", "x", "y", "z"} <= keys and all(is_num(d[k]) for k in "twxyz"):
            quats.append((d["t"], d["w"], d["x"], d["y"], d["z"]))

    def clean(rows):  # sort + drop duplicated timestamps
        out, seen = [], set()
        for r in sorted(rows, key=lambda r: r[0]):
            if r[0] not in seen:
                seen.add(r[0])
                out.append(r)
        return out

    return {k: clean(v) for k, v in series.items()}, clean(quats)


def quat_to_euler_zyx(w, x, y, z):
    """Hamilton quaternion -> (roll, pitch, yaw) in degrees, ZYX convention.
    DJI does not document the frame or the direction of the rotation
    (body->world or the opposite): validate on a known motion."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {path.name}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mp4")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--dump-json", action="store_true", help="write all the decoded telemetry to JSON")
    a = ap.parse_args()

    src = Path(a.mp4)
    out = Path(a.outdir) if a.outdir else src.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    tp = telemetry_parser.Parser(str(src))
    print(f"Camera: {tp.camera}  Model: {tp.model}")

    telemetry = tp.telemetry()
    if a.dump_json:
        p = out / f"{stem}_telemetry.json"
        p.write_text(json.dumps(telemetry, default=str, indent=1))
        print(f"  {p.name} written")

    series, quats = collect(telemetry)

    # 1) Raw data (units and sensor frame as decoded; t is the timestamp stored
    #    in the file, usually in ms: check it against the clip duration)
    if series["Gyroscope"]:
        write_csv(out / f"{stem}_gyro_raw.csv", ["t", "gx", "gy", "gz"], series["Gyroscope"])
    if series["Accelerometer"]:
        write_csv(out / f"{stem}_accl_raw.csv", ["t", "ax", "ay", "az"], series["Accelerometer"])
    if series["Magnetometer"]:
        write_csv(out / f"{stem}_magn_raw.csv", ["t", "mx", "my", "mz"], series["Magnetometer"])

    # 2) IMU normalized by telemetry-parser (Gyroflow camera frame, gyro in deg/s;
    #    accelerometer: check the norm at rest, ~1 if in g, ~9.81 if in m/s²)
    norm = tp.normalized_imu() or []
    rows = []
    for s in norm:
        g = s.get("gyro") or [None] * 3
        acc = s.get("accl") or [None] * 3
        m = s.get("magn") or [None] * 3
        rows.append([s.get("timestamp_ms"), *g, *acc, *m])
    if rows:
        write_csv(out / f"{stem}_imu_norm.csv",
                  ["timestamp_ms", "gx_dps", "gy_dps", "gz_dps", "ax", "ay", "az", "mx", "my", "mz"], rows)

    # 3) Attitude: quaternions from the camera's own sensor fusion
    if quats:
        rows = [[t, w, x, y, z, *quat_to_euler_zyx(w, x, y, z)] for t, w, x, y, z in quats]
        write_csv(out / f"{stem}_attitude.csv",
                  ["t", "qw", "qx", "qy", "qz", "roll_deg", "pitch_deg", "yaw_deg"], rows)
    else:
        print("  No quaternion found: run again with --dump-json to inspect, "
              "or use djmd_inspect.py.")

    if not (series["Gyroscope"] or norm or quats):
        print("No IMU data: check that the video was recorded with stabilization off "
              "and the FOV set to Wide.")


if __name__ == "__main__":
    main()

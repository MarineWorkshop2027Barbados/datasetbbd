#!/usr/bin/env python3
"""
Merge the Action 5 Pro bag and the Deep Trekker ROV log into one bag, on the
ROV clock, using the offset and drift found by sync_rov_camera.py.

    pip install rosbags numpy

    python merge_bag.py bags/DJI_xxxx rov_20261113_173335.csv --sync sync/sync.json -o merged
    python merge_bag.py bags/DJI_xxxx rov.csv --sync sync/sync.json -o merged.bag --ros 1

Camera topics (from mp4_to_rosbag.py) are copied with their stamps moved to
the ROV clock:  t_rov = t_cam + offset_at_clip_start + drift * (t_cam - t0_cam).

ROV topics (frame conventions of the Deep Trekker state):
  /rov/orientation     geometry_msgs/QuaternionStamped  body FRD -> local NED, from heading/pitch/roll
  /rov/attitude_deg    geometry_msgs/Vector3Stamped     x = roll, y = pitch, z = heading (deg, as logged)
  /rov/depth           geometry_msgs/PointStamped       z = depth (m, positive down)
  /rov/altitude        geometry_msgs/PointStamped       z = DVL altitude above the seabed (m), valid samples only
  /rov/dvl/velocity    geometry_msgs/TwistStamped       linear x/y/z = forward/rightward/downward (m/s), valid only
  /rov/usbl/position   geometry_msgs/PointStamped       x, y = USBL local position (as logged)
  /rov/water_temperature  sensor_msgs/Temperature       (°C)
  /rov/raw/<column>    std_msgs/Float64                 every other numeric CSV column, written when it changes
                                                        (booleans as 0/1)

By default only the ROV samples within the camera clip (± --margin s) are kept;
--all-rov keeps the whole log.
"""
import argparse
import dataclasses
import heapq
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rov_analysis import load_rov_csv  # noqa: E402

TYPED_KEYS = {"t", "heading", "pitch", "roll", "depth", "altitude", "altitude_valid", "vel_forward",
              "vel_right", "vel_down", "vel_valid", "x", "y", "water_temp"}

# ---------------------------------------------------------------- time mapping

class ClockMap:
    """Camera clock -> ROV clock, from sync.json."""

    def __init__(self, sync, use_drift=True):
        self.offset = sync.get("offset_at_clip_start_s", sync["offset_s"])
        self.t0 = sync["camera_t0_old"]
        drift = sync.get("drift_ppm") if use_drift else None
        self.slope = (drift or 0.0) * 1e-6

    def __call__(self, t_cam_ns):
        t = t_cam_ns * 1e-9
        return int(round((t + self.offset + self.slope * (t - self.t0)) * 1e9))

# ---------------------------------------------------------------- ROV messages

def quat_from_heading_pitch_roll(heading, pitch, roll):
    """(w, x, y, z) of R = Rz(heading) Ry(pitch) Rx(roll), degrees: body FRD -> local NED."""
    y, p, r = (math.radians(v) / 2 for v in (heading, pitch, roll))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def topic_for_column(col):
    name = re.sub(r"[^A-Za-z0-9_/]", "_", col.replace(".", "/")).strip("/")
    name = re.sub(r"/+", "/", name)
    name = "/".join(("_" + s if s[:1].isdigit() else s) for s in name.split("/") if s)
    return f"/rov/raw/{name}"


def rov_messages(data, mapping, raw, t_min, t_max):
    """Yields (t_ns, topic, kind, payload) for the ROV rows in [t_min, t_max]."""
    t = data["t"]
    sel = np.where((t >= t_min) & (t <= t_max))[0]

    def val(key, i):
        v = data.get(key)
        return None if v is None or not np.isfinite(v[i]) else float(v[i])

    out = []
    for i in sel:
        t_ns = int(round(t[i] * 1e9))
        h, p, r = val("heading", i), val("pitch", i), val("roll", i)
        if None not in (h, p, r):
            out.append((t_ns, "/rov/orientation", "quaternion", quat_from_heading_pitch_roll(h, p, r)))
            out.append((t_ns, "/rov/attitude_deg", "vector3", (r, p, h)))
        d = val("depth", i)
        if d is not None:
            out.append((t_ns, "/rov/depth", "point", (0.0, 0.0, d)))
        alt, alt_ok = val("altitude", i), val("altitude_valid", i)
        if alt is not None and (alt_ok is None or alt_ok > 0.5):
            out.append((t_ns, "/rov/altitude", "point", (0.0, 0.0, alt)))
        vf, vr, vd, v_ok = val("vel_forward", i), val("vel_right", i), val("vel_down", i), val("vel_valid", i)
        if None not in (vf, vr, vd) and (v_ok is None or v_ok > 0.5):
            out.append((t_ns, "/rov/dvl/velocity", "twist", (vf, vr, vd)))
        x, y = val("x", i), val("y", i)
        if x is not None and y is not None:
            out.append((t_ns, "/rov/usbl/position", "point", (x, y, 0.0)))
        wt = val("water_temp", i)
        if wt is not None:
            out.append((t_ns, "/rov/water_temperature", "temperature", wt))

    # every other numeric column, only when its value changes
    typed_cols = {mapping[k] for k in TYPED_KEYS if k in mapping}
    for col, v in raw.items():
        if col in typed_cols:
            continue
        topic, last = topic_for_column(col), None
        for i in sel:
            if np.isfinite(v[i]) and v[i] != last:
                out.append((int(round(t[i] * 1e9)), topic, "float64", float(v[i])))
                last = v[i]
    out.sort(key=lambda m: m[0])
    return out

# ---------------------------------------------------------------- bag writing

class Writer:
    def __init__(self, path, ros, storage):
        from rosbags.typesys import Stores, get_typestore
        self.ros = ros
        self.ts = get_typestore(Stores.ROS1_NOETIC if ros == 1 else Stores.ROS2_HUMBLE)
        self.T = self.ts.types
        if ros == 1:
            from rosbags.rosbag1 import Writer as W1
            self.w = W1(Path(path))
        else:
            from rosbags.rosbag2 import Writer as W2
            try:
                from rosbags.rosbag2 import StoragePlugin
                plugin = StoragePlugin.MCAP if storage == "mcap" else StoragePlugin.SQLITE3
                self.w = W2(Path(path), version=8, storage_plugin=plugin)
            except (ImportError, TypeError):  # older rosbags versions
                self.w = W2(Path(path))
        self.w.open()
        self.conns, self.seq, self.count = {}, {}, 0

    def conn(self, topic, msgtype):
        if topic not in self.conns:
            self.conns[topic] = self.w.add_connection(topic, msgtype, typestore=self.ts)
            self.seq[topic] = 0
        return self.conns[topic]

    def header(self, topic, t_ns, frame_id):
        stamp = self.T["builtin_interfaces/msg/Time"](sec=int(t_ns // 1_000_000_000), nanosec=int(t_ns % 1_000_000_000))
        Header = self.T["std_msgs/msg/Header"]
        if self.ros == 1:
            self.seq[topic] = self.seq.get(topic, 0) + 1
            return Header(seq=self.seq[topic], stamp=stamp, frame_id=frame_id)
        return Header(stamp=stamp, frame_id=frame_id)

    def write(self, topic, msg, t_ns):
        c = self.conn(topic, msg.__msgtype__)
        serialize = self.ts.serialize_ros1 if self.ros == 1 else self.ts.serialize_cdr
        self.w.write(c, t_ns, serialize(msg, msg.__msgtype__))
        self.count += 1

    def write_rov(self, t_ns, topic, kind, payload):
        T = self.T
        if kind == "float64":
            self.conn(topic, "std_msgs/msg/Float64")
            self.write(topic, T["std_msgs/msg/Float64"](data=payload), t_ns)
            return
        frame = {"/rov/orientation": "rov_ned", "/rov/usbl/position": "rov_usbl_local"}.get(topic, "rov_body_frd")
        h = self.header(topic, t_ns, frame)
        if kind == "quaternion":
            w, x, y, z = payload
            msg = T["geometry_msgs/msg/QuaternionStamped"](
                header=h, quaternion=T["geometry_msgs/msg/Quaternion"](x=x, y=y, z=z, w=w))
        elif kind == "vector3":
            msg = T["geometry_msgs/msg/Vector3Stamped"](header=h, vector=T["geometry_msgs/msg/Vector3"](*payload))
        elif kind == "point":
            msg = T["geometry_msgs/msg/PointStamped"](header=h, point=T["geometry_msgs/msg/Point"](*payload))
        elif kind == "twist":
            V = T["geometry_msgs/msg/Vector3"]
            msg = T["geometry_msgs/msg/TwistStamped"](
                header=h, twist=T["geometry_msgs/msg/Twist"](linear=V(*payload), angular=V(0.0, 0.0, 0.0)))
        elif kind == "temperature":
            msg = T["sensor_msgs/msg/Temperature"](header=h, temperature=payload, variance=0.0)
        else:
            raise ValueError(kind)
        self.write(topic, msg, t_ns)

    def write_camera(self, topic, msg, t_ns):
        """Copies a camera message with its header stamp moved to t_ns."""
        if hasattr(msg, "header"):
            msg = dataclasses.replace(msg, header=self.header(topic, t_ns, msg.header.frame_id))
        self.write(topic, msg, t_ns)

    def close(self):
        self.w.close()

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camera_bag", help="bag written by mp4_to_rosbag.py")
    ap.add_argument("rov_csv", help="ROV log written by log_rov_csv.py")
    ap.add_argument("--sync", required=True, help="sync.json written by sync_rov_camera.py")
    ap.add_argument("-o", "--out", required=True, help="output folder (ROS 2) or .bag file (ROS 1)")
    ap.add_argument("--ros", type=int, choices=(1, 2), default=2)
    ap.add_argument("--storage", choices=("mcap", "sqlite3"), default="mcap", help="ROS 2 only")
    ap.add_argument("--map", action="append", default=[], metavar="KEY=COLUMN", help="force a ROV column")
    ap.add_argument("--margin", type=float, default=10.0, help="ROV data kept before / after the clip (s)")
    ap.add_argument("--all-rov", action="store_true", help="keep the whole ROV log")
    ap.add_argument("--no-drift", action="store_true", help="ignore the clock drift found by the sync")
    ap.add_argument("--no-raw", action="store_true", help="don't write the /rov/raw/* topics")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        if not a.overwrite:
            sys.exit(f"{out} already exists: delete it or run again with --overwrite")
        shutil.rmtree(out) if out.is_dir() else out.unlink()

    sync = json.loads(Path(a.sync).read_text())
    clock = ClockMap(sync, use_drift=not a.no_drift)
    print(f"Clock map: t_rov = t_cam + {clock.offset:.3f} s"
          + (f" + {clock.slope * 1e6:+.0f} ppm" if clock.slope else "")
          + f"  (sync quality: {sync.get('quality', '?')})")
    if sync.get("quality") not in (None, "good"):
        print("  WARNING: the synchronization was flagged as uncertain, check sync.png")

    data, mapping, raw = load_rov_csv(a.rov_csv, dict(m.split("=", 1) for m in a.map))
    if a.no_raw:
        raw = {}

    from view_rosbag import open_reader, stamp_ns
    w = Writer(out, a.ros, a.storage)
    n_cam = 0
    with open_reader(a.camera_bag) as reader:
        t_lo = clock(reader.start_time) * 1e-9
        t_hi = clock(reader.end_time) * 1e-9
        rng = (-math.inf, math.inf) if a.all_rov else (t_lo - a.margin, t_hi + a.margin)
        rov = rov_messages(data, mapping, raw, *rng)
        print(f"Camera bag: {reader.duration * 1e-9:.1f} s; ROV messages kept: {len(rov)}")

        def camera_stream():
            for conn, t_bag, raw_msg in reader.messages():
                msg = reader.deserialize(raw_msg, conn.msgtype)
                t_src = stamp_ns(msg) if hasattr(msg, "header") else t_bag
                yield clock(t_src), 0, conn.topic, msg

        def rov_stream():
            for t_ns, topic, kind, payload in rov:
                yield t_ns, 1, topic, (kind, payload)

        for t_ns, source, topic, item in heapq.merge(camera_stream(), rov_stream(), key=lambda m: (m[0], m[1])):
            if source == 0:
                w.write_camera(topic, item, t_ns)
                n_cam += 1
                if n_cam % 2000 == 0:
                    print(f"  {n_cam} camera messages", end="\r")
            else:
                w.write_rov(t_ns, topic, *item)
    w.close()

    print(f"\nDone: {w.count} messages ({n_cam} camera, {w.count - n_cam} ROV) on {len(w.conns)} topics -> {out}")
    for topic in sorted(w.conns):
        print(f"  {topic}")


if __name__ == "__main__":
    main()

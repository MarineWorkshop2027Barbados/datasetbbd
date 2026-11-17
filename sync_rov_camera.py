#!/usr/bin/env python3
"""
Synchronize the Action 5 Pro (mounted on the ROV) with the ROV log, by
matching the rotations seen by both.

The angular speed |ω| does not depend on how the camera is mounted on the
ROV, so the clock offset is found by cross-correlating |ω| from the ROV
attitude (heading / pitch / roll in the CSV written by log_rov_csv.py) with
|ω| from the camera attitude (/action5/imu in the bag). Once synchronized,
the camera mounting rotation (ROV body -> camera IMU) is estimated by
aligning the relative rotations (hand-eye).

    pip install rosbags numpy matplotlib

    python sync_rov_camera.py rov_20261113_173335.csv bags/DJI_xxxx -o sync/
    python sync_rov_camera.py rov.csv bags/DJI_xxxx --map heading=nav_heading   # force a column

Outputs (in -o):
  sync.json   offset (t_rov = t_cam + offset), correlation quality, clock drift,
              mounting rotation; read by merge_bag.py
  sync.png    |ω| of both sensors after alignment + correlation curve

Tip: at the start of each dive, give the ROV a few sharp yaw turns (and
tilts if possible) while both are recording: a distinctive motion makes the
correlation peak unambiguous. A ROV holding heading and depth gives a flat
|ω| and an unreliable offset.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rov_analysis import load_rov_csv  # noqa: E402

# ---------------------------------------------------------------- rotations

def q2R(q):
    """q: (N,4) as (w,x,y,z) -> (N,3,3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rot_angle(R):
    return np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def rot_log(R):
    """(N,3,3) -> rotation vectors (N,3)."""
    th = rot_angle(R)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    out = np.zeros_like(v)
    small = th < 1e-6
    big = th > math.pi - 1e-3
    mid = ~small & ~big
    out[mid] = v[mid] * (th[mid] / (2 * np.sin(th[mid])))[:, None]
    out[small] = v[small] / 2
    for i in np.where(big)[0]:  # close to pi: axis taken from (R + I) / 2
        B = (R[i] + np.eye(3)) / 2
        k = np.argmax(np.diag(B))
        axis = B[:, k] / math.sqrt(max(B[k, k], 1e-12))
        out[i] = axis / np.linalg.norm(axis) * th[i]
    return out


def kabsch(A, B):
    """Rotation X minimizing sum |a_i - X b_i|^2."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

# ---------------------------------------------------------------- inputs

def load_rov(path, overrides):
    """ROV attitude: (t_unix s, heading, pitch, roll in deg) for the rows where all are defined."""
    data, mapping, _ = load_rov_csv(path, overrides)
    for k in ("heading", "pitch", "roll"):
        if k not in data:
            sys.exit(f"No '{k}' column found: use --map {k}=COLUMN")
    t, h, p, r = data["t"], data["heading"], data["pitch"], data["roll"]
    ok = np.isfinite(t) & np.isfinite(h) & np.isfinite(p) & np.isfinite(r)
    t, h, p, r = t[ok], h[ok], p[ok], r[ok]
    if len(t) < 10:
        sys.exit("Fewer than 10 usable ROV rows")
    print(f"ROV: {len(t)} samples over {t[-1] - t[0]:.1f} s (~{(len(t) - 1) / (t[-1] - t[0]):.1f} Hz), "
          f"columns: {mapping['t']}, {mapping['heading']}, {mapping['pitch']}, {mapping['roll']}")
    return t, h, p, r


def load_camera(bag):
    """Camera attitude from /action5/imu: (t s, quaternions w,x,y,z, stamp of the first video frame)."""
    from view_rosbag import IMG_TOPIC, IMU_TOPIC, open_reader, stamp_ns
    t_ns, q, t0_ns = [], [], None
    with open_reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            if m.orientation_covariance[0] == -1:
                continue
            o = m.orientation
            t_ns.append(stamp_ns(m))
            q.append((o.w, o.x, o.y, o.z))
        for conn, _, raw in reader.messages(connections=[c for c in reader.connections if c.topic == IMG_TOPIC]):
            t0_ns = stamp_ns(reader.deserialize(raw, conn.msgtype))
            break
    if not t_ns:
        sys.exit("No orientation in /action5/imu")
    order = np.argsort(t_ns)
    t = np.asarray(t_ns, dtype=np.int64)[order] * 1e-9
    q = np.asarray(q)[order]
    print(f"Camera: {len(t)} attitude samples over {t[-1] - t[0]:.1f} s (~{(len(t) - 1) / (t[-1] - t[0]):.0f} Hz)")
    return t, q, (t0_ns * 1e-9 if t0_ns is not None else t[0])

# ---------------------------------------------------------------- signals

def euler_zyx_to_R(yaw_deg, pitch_deg, roll_deg):
    """R = Rz(yaw) Ry(pitch) Rx(roll), angles in degrees, arrays of length N."""
    y, p, r = (np.radians(np.asarray(v, float)) for v in (yaw_deg, pitch_deg, roll_deg))
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    return np.stack([
        np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
        np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
        np.stack([-sp, cp * sr, cp * cr], -1),
    ], -2)


def rov_rotations(t, heading, pitch, roll, grid):
    """ROV rotations on a time grid (heading unwrapped before interpolation)."""
    h = np.degrees(np.unwrap(np.radians(heading)))
    return euler_zyx_to_R(np.interp(grid, t, h), np.interp(grid, t, pitch), np.interp(grid, t, roll))


def cam_rotations(t, q, grid):
    """Camera rotations on a time grid (nearest sample: the IMU runs at 200 Hz or more)."""
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    idx = np.where(np.abs(t[idx - 1] - grid) < np.abs(t[idx] - grid), idx - 1, idx)
    return q2R(q[idx])


def angular_speed(R, dt):
    """|ω| (rad/s) between consecutive rotations: frame independent."""
    w = rot_angle(np.transpose(R[:-1], (0, 2, 1)) @ R[1:]) / dt
    return np.append(w, w[-1])


def ncc_search(template, signal):
    """Normalized cross-correlation of a template slid along a longer signal.
    Returns ncc[L] for template[i] <-> signal[i + L], L = 0 .. len(signal) - len(template)."""
    m = len(template)
    tz = template - template.mean()
    tn = np.sqrt((tz ** 2).sum()) or 1.0
    num = np.correlate(signal, tz, mode="valid")
    c1 = np.concatenate([[0], np.cumsum(signal)])
    c2 = np.concatenate([[0], np.cumsum(signal ** 2)])
    s1 = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    var = np.maximum(s2 - s1 ** 2 / m, 1e-12)
    return num / (tn * np.sqrt(var))


def subsample_peak(c, k):
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        if den < 0:
            return k + 0.5 * (c[k - 1] - c[k + 1]) / den
    return float(k)


def estimate_offset(t_rov, att_rov, t_cam, q_cam, rate, smooth_s):
    """Global offset (t_rov = t_cam + offset) by NCC of |ω|, whatever the clock gap."""
    dt = 1.0 / rate
    g_rov = np.arange(t_rov[0], t_rov[-1], dt)
    g_cam = np.arange(t_cam[0], t_cam[-1], dt)
    w_rov = angular_speed(rov_rotations(t_rov, *att_rov, g_rov), dt)
    w_cam = angular_speed(cam_rotations(t_cam, q_cam, g_cam), dt)
    k = max(1, int(round(smooth_s * rate)))
    box = np.ones(k) / k
    w_rov, w_cam = np.convolve(w_rov, box, "same"), np.convolve(w_cam, box, "same")

    cam_is_template = len(w_cam) <= len(w_rov)
    template, signal = (w_cam, w_rov) if cam_is_template else (w_rov, w_cam)
    if len(template) < 10 * rate:
        sys.exit("Overlap too short (< 10 s) for a reliable correlation")
    c = ncc_search(template, signal)
    best = int(np.argmax(c))
    lag = subsample_peak(c, best)
    # second best peak, at least 5 s away
    mask = np.abs(np.arange(len(c)) - best) > 5 * rate
    second = float(c[mask].max()) if mask.any() else float("nan")
    if cam_is_template:
        offset = g_rov[0] + lag * dt - g_cam[0]
    else:
        offset = g_rov[0] - (g_cam[0] + lag * dt)
    return dict(offset=offset, ncc=float(c[best]), second=second, curve=c,
                curve_lag0=(g_rov[0] - g_cam[0]) if cam_is_template else (g_rov[0] - g_cam[0]),
                cam_is_template=cam_is_template, dt=dt,
                g_rov=g_rov, w_rov=w_rov, g_cam=g_cam, w_cam=w_cam)


def local_offsets(t_rov, att_rov, t_cam, q_cam, offset, rate, smooth_s, window_s, search_s):
    """Offset measured on successive windows of the camera clip: reveals clock drift."""
    out = []
    t = t_cam[0]
    while t + window_s <= t_cam[-1]:
        sel_c = (t_cam >= t) & (t_cam <= t + window_s)
        lo, hi = t + offset - search_s, t + window_s + offset + search_s
        sel_r = (t_rov >= lo - 1) & (t_rov <= hi + 1)
        if sel_r.sum() > 5 and sel_c.sum() > 10:
            try:
                r = estimate_offset(t_rov[sel_r], tuple(a[sel_r] for a in att_rov),
                                    t_cam[sel_c], q_cam[sel_c], rate, smooth_s)
                if abs(r["offset"] - offset) <= search_s:
                    out.append((t + window_s / 2, r["offset"], r["ncc"]))
            except SystemExit:
                pass
        t += window_s / 2
    return out


def mounting_rotation(t_rov, att_rov, t_cam, q_cam, offset, rate, delta_s):
    """ROV body -> camera IMU rotation X, from relative rotations over delta_s.
    The DJI quaternion direction is unknown: both are tried, the best is kept."""
    lo, hi = max(t_cam[0] + offset, t_rov[0]), min(t_cam[-1] + offset, t_rov[-1])
    if hi - lo < 3 * delta_s:
        return None
    grid = np.arange(lo, hi, 1.0 / rate)
    R_rov = rov_rotations(t_rov, *att_rov, grid)
    R_cam = cam_rotations(t_cam, q_cam, grid - offset)
    step = max(1, int(round(delta_s * rate)))
    i, k = np.arange(len(grid) - step), np.arange(step, len(grid))
    B = np.transpose(R_rov[i], (0, 2, 1)) @ R_rov[k]
    best = None
    for inverse in (False, True):
        C = np.transpose(R_cam, (0, 2, 1)) if inverse else R_cam
        A = np.transpose(C[i], (0, 2, 1)) @ C[k]
        moving = rot_angle(B) > math.radians(1.0)  # pairs with enough rotation
        if moving.sum() < 10:
            return None
        X = kabsch(rot_log(A[moving]), rot_log(B[moving]))
        err = rot_angle(np.transpose(A[moving], (0, 2, 1)) @ (X @ B[moving] @ X.T))
        if best is None or np.median(err) < best["median_err"]:
            best = dict(X=X, inverse=inverse, median_err=float(np.median(err)), n_pairs=int(moving.sum()))
    return best


def fmt_rot(X):
    rv = rot_log(X[None])[0]
    angle = np.degrees(np.linalg.norm(rv))
    axis = rv / (np.linalg.norm(rv) or 1)
    return f"{angle:.1f} deg about [{axis[0]:+.3f} {axis[1]:+.3f} {axis[2]:+.3f}]"

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rov_csv", help="CSV written by log_rov_csv.py")
    ap.add_argument("bag", help="bag written by mp4_to_rosbag.py")
    ap.add_argument("-o", "--outdir", default="sync")
    ap.add_argument("--map", action="append", default=[], metavar="KEY=COLUMN",
                    help="force a ROV column (keys t, heading, pitch, roll), see rov_analysis.py")
    ap.add_argument("--rate", type=float, default=10.0, help="analysis rate (Hz)")
    ap.add_argument("--smooth", type=float, default=0.3, help="|ω| smoothing window (s)")
    ap.add_argument("--window", type=float, default=60.0, help="window for the drift estimate (s)")
    ap.add_argument("--delta", type=float, default=1.0, help="interval of the relative rotations (s)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t_rov, heading, pitch, roll = load_rov(a.rov_csv, dict(m.split("=", 1) for m in a.map))
    att_rov = (heading, pitch, roll)
    t_cam, q_cam, t0_cam = load_camera(a.bag)

    rov_rate = (len(t_rov) - 1) / (t_rov[-1] - t_rov[0])
    if rov_rate < a.rate / 2:
        print(f"  note: ROV log at ~{rov_rate:.1f} Hz, the offset precision is limited to ~{0.2 / rov_rate:.2f} s")

    # 1) global offset
    r = estimate_offset(t_rov, att_rov, t_cam, q_cam, a.rate, a.smooth)
    offset = r["offset"]
    nominal = t_rov[0] - t_cam[0]
    print(f"\nOffset: t_rov = t_cam + {offset:.3f} s   (NCC peak {r['ncc']:.3f}, next peak {r['second']:.3f})")
    hours = offset / 3600
    if abs(offset) > 600 and abs(hours - round(hours)) * 3600 < 120:
        print(f"  = {round(hours):+d} h {offset - round(hours) * 3600:+.1f} s: the hours are a time-zone gap "
              "(the MP4 creation time is probably local time)")
    quality = "good" if r["ncc"] > 0.6 and (math.isnan(r["second"]) or r["ncc"] - r["second"] > 0.15) else "UNCERTAIN"
    print(f"  correlation quality: {quality}")
    if quality != "good":
        print("  -> the motion is not distinctive enough: check sync.png, or record sharp yaw turns at the start")

    # 2) drift
    local = local_offsets(t_rov, att_rov, t_cam, q_cam, offset, a.rate, a.smooth, a.window, search_s=5.0)
    drift_ppm, offset_start = None, offset
    good = [(t, o) for t, o, c in local if c > 0.5]
    if len(good) >= 3:
        tt, oo = np.array(good).T
        slope, intercept = np.polyfit(tt - t_cam[0], oo, 1)
        drift_ppm = slope * 1e6
        offset_start = intercept + slope * (t0_cam - t_cam[0])  # offset at the first video frame
        print(f"Drift: {drift_ppm:+.0f} ppm ({slope * 3600:+.3f} s per hour) over {len(good)} windows, "
              f"spread {np.std(oo - (slope * (tt - t_cam[0]) + intercept)) * 1e3:.0f} ms")
    else:
        print("Drift: not estimated (not enough distinctive windows)")

    # 3) mounting rotation
    mount = mounting_rotation(t_rov, att_rov, t_cam, q_cam, offset, a.rate, a.delta)
    if mount:
        print(f"Mounting rotation ROV -> camera IMU: {fmt_rot(mount['X'])}, residual "
              f"{np.degrees(mount['median_err']):.2f} deg (median, {mount['n_pairs']} pairs), "
              f"DJI quaternion {'world -> body' if mount['inverse'] else 'body -> world'}")
    else:
        print("Mounting rotation: not estimated (not enough rotation)")

    t0_new = t0_cam + offset_start
    if drift_ppm is not None and abs(drift_ppm) > 20:
        print(f"  note: with this drift, the camera clock gains {abs(drift_ppm) * 1e-6 * (t_cam[-1] - t_cam[0]) * 1e3:.0f} ms "
              "over the clip; --t0 aligns the start of the clip")
    print(f"\nNext step: merge everything on the ROV clock\n  python merge_bag.py {a.bag} {a.rov_csv} "
          f"--sync {out / 'sync.json'} -o merged")

    result = dict(offset_s=offset, offset_at_clip_start_s=offset_start, convention="t_rov = t_cam + offset_s", ncc_peak=r["ncc"],
                  ncc_second_peak=r["second"], quality=quality, nominal_start_gap_s=nominal,
                  drift_ppm=drift_ppm, local_offsets=[dict(t_cam=t, offset_s=o, ncc=c) for t, o, c in local],
                  camera_t0_old=t0_cam, camera_t0_new=t0_new,
                  mounting=None if not mount else dict(R_rov_to_cam=mount["X"].tolist(),
                                                       angle_axis=fmt_rot(mount["X"]),
                                                       median_residual_deg=float(np.degrees(mount["median_err"])),
                                                       dji_quaternion="world->body" if mount["inverse"] else "body->world"))
    (out / "sync.json").write_text(json.dumps(result, indent=2))
    print(f"-> {out / 'sync.json'}")

    # 4) figure
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5))
    ax[0].plot(r["g_rov"] - r["g_rov"][0], np.degrees(r["w_rov"]), lw=0.8, label="ROV |ω|")
    ax[0].plot(r["g_cam"] + offset - r["g_rov"][0], np.degrees(r["w_cam"]), lw=0.8, alpha=0.8,
               label="camera |ω| (shifted)")
    lo, hi = t_cam[0] + offset - r["g_rov"][0], t_cam[-1] + offset - r["g_rov"][0]
    ax[0].set_xlim(lo - 10, hi + 10)
    ax[0].set_xlabel("time since the start of the ROV log (s)")
    ax[0].set_ylabel("|ω| (deg/s)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    lags = np.arange(len(r["curve"])) * r["dt"]
    ax[1].plot(lags, r["curve"], lw=0.8)
    ax[1].set_xlabel("lag along the longer signal (s)")
    ax[1].set_ylabel("NCC")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "sync.png", dpi=150)
    print(f"-> {out / 'sync.png'}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()

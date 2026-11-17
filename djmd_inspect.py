#!/usr/bin/env python3
"""
Read the 'djmd' metadata track of a DJI MP4 directly, with no dependency
(Python standard library only) and no protobuf schema.

Useful to check what telemetry-parser returns, or as a fallback when your
firmware is not supported yet.

1) Explore the structure, find IMU and quaternion candidates:
     python djmd_inspect.py DJI_xxxx.MP4
2) Export a sub-message (one row per occurrence, all its scalar fields as
   columns), with the timestamp of the video frame:
     python djmd_inspect.py DJI_xxxx.MP4 --export 3/2/10 -o accel.csv
3) Look for depth / pressure over the whole clip (use a dive clip with a
   descent and an ascent):
     python djmd_inspect.py DJI_xxxx.MP4 --scan --scan-csv scalars.csv

Paths (e.g. 3/2/10) are nested protobuf field numbers; step 1 prints them.
A scalar field is written PATH:FIELD (e.g. 3/2/13:3).

Known fields on the Osmo Action 5 Pro (dvtm_ac204, cf. ExifTool DJI tags):
  3/1:1 frame number, 3/1:2 timestamp (µs), 3/2/3:1 ISO, 3/2/6:1 color temperature,
  3/2/9 per-frame quaternion, 3/2/10:2-4 accelerometer (g, one sample per frame),
  3/3/2/3 attitude quaternion (~1 kHz), 3/3/2:1 its timestamp (µs),
  3/2/13:3 pressure (hPa), 3/2/13:2 temperature (hundredths of °C, assumed).
"""
import argparse
import csv
import math
import struct
import sys
from collections import defaultdict

# ---------------------------------------------------------------- MP4 / ISO BMFF

def iter_boxes(f, start, end):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, typ = struct.unpack(">I4s", f.read(8))
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            break
        yield typ.decode("latin1"), pos + hdr, pos + size
        pos += size


def child(f, start, end, name):
    for typ, s, e in iter_boxes(f, start, end):
        if typ == name:
            return s, e
    return None


def read_at(f, start, end):
    f.seek(start)
    return f.read(end - start)


def parse_track(f, trak_start, trak_end):
    """Returns dict(format, handler, timescale, samples=[(offset, size, time_s)])."""
    mdia = child(f, trak_start, trak_end, "mdia")
    if not mdia:
        return None
    hdlr = child(f, *mdia, "hdlr")
    handler = read_at(f, *hdlr)[8:12].decode("latin1") if hdlr else "?"
    mdhd = read_at(f, *child(f, *mdia, "mdhd"))
    timescale = struct.unpack(">I", mdhd[20:24] if mdhd[0] == 1 else mdhd[12:16])[0]
    minf = child(f, *mdia, "minf")
    stbl = child(f, *minf, "stbl") if minf else None
    if not stbl:
        return None
    boxes = {typ: read_at(f, s, e) for typ, s, e in iter_boxes(f, *stbl)}
    fmt = boxes["stsd"][12:16].decode("latin1") if "stsd" in boxes else "?"

    # sample sizes
    stsz = boxes["stsz"]
    fixed, n = struct.unpack(">II", stsz[4:12])
    sizes = [fixed] * n if fixed else list(struct.unpack(f">{n}I", stsz[12:12 + 4 * n]))
    # chunk offsets
    if "stco" in boxes:
        m = struct.unpack(">I", boxes["stco"][4:8])[0]
        chunks = struct.unpack(f">{m}I", boxes["stco"][8:8 + 4 * m])
    else:
        m = struct.unpack(">I", boxes["co64"][4:8])[0]
        chunks = struct.unpack(f">{m}Q", boxes["co64"][8:8 + 8 * m])
    # samples per chunk
    k = struct.unpack(">I", boxes["stsc"][4:8])[0]
    stsc = [struct.unpack(">III", boxes["stsc"][8 + 12 * i:20 + 12 * i]) for i in range(k)]
    offsets, si = [], 0
    for ci in range(len(chunks)):
        per_chunk = next(spc for first, spc, _ in reversed(stsc) if ci + 1 >= first)
        off = chunks[ci]
        for _ in range(per_chunk):
            if si >= n:
                break
            offsets.append(off)
            off += sizes[si]
            si += 1
    # sample times
    k = struct.unpack(">I", boxes["stts"][4:8])[0]
    times, t = [], 0
    for i in range(k):
        count, delta = struct.unpack(">II", boxes["stts"][8 + 8 * i:16 + 8 * i])
        for _ in range(count):
            times.append(t / timescale)
            t += delta
    return dict(format=fmt, handler=handler, timescale=timescale,
                samples=list(zip(offsets, sizes, times)))


def load_tracks(path):
    f = open(path, "rb")
    f.seek(0, 2)
    size = f.tell()
    moov = child(f, 0, size, "moov")
    if not moov:
        sys.exit("No moov box: truncated file or not an MP4?")
    tracks = [parse_track(f, s, e) for typ, s, e in iter_boxes(f, *moov) if typ == "trak"]
    return f, [t for t in tracks if t]

# ---------------------------------------------------------------- raw protobuf

def varint(b, i):
    result = shift = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return result, i


def parse_pb(b):
    """Schema-less protobuf decoding. Returns [(field, wiretype, value)] or
    None if the bytes are not a valid message."""
    out, i, n = [], 0, len(b)
    try:
        while i < n:
            key, i = varint(b, i)
            fnum, wt = key >> 3, key & 7
            if fnum == 0:
                return None
            if wt == 0:
                v, i = varint(b, i)
            elif wt == 1:
                v, i = b[i:i + 8], i + 8
            elif wt == 2:
                length, i = varint(b, i)
                v, i = b[i:i + length], i + length
            elif wt == 5:
                v, i = b[i:i + 4], i + 4
            else:
                return None
            if i > n:
                return None
            out.append((fnum, wt, v))
    except IndexError:
        return None
    return out


def looks_text(b):
    return len(b) > 0 and all(32 <= c < 127 or c in (9, 10, 13) for c in b)


def decode(b, depth=0):
    """Tree: list of (field, kind, value); kind in msg/int/f32/f64/str/bytes."""
    nodes = []
    for fnum, wt, v in parse_pb(b) or []:
        if wt == 0:
            nodes.append((fnum, "int", v))
        elif wt == 5:
            nodes.append((fnum, "f32", struct.unpack("<f", v)[0]))
        elif wt == 1:
            nodes.append((fnum, "f64", struct.unpack("<d", v)[0]))
        else:
            sub = parse_pb(v) if depth < 12 and not looks_text(v) else None
            if sub:
                nodes.append((fnum, "msg", decode(v, depth + 1)))
            elif looks_text(v):
                nodes.append((fnum, "str", v.decode("ascii")))
            else:
                nodes.append((fnum, "bytes", v))
    return nodes


def iter_msgs(nodes, path=""):
    """Yields (path, list of scalars) for every sub-message."""
    scalars = [(f, k, v) for f, k, v in nodes if k != "msg"]
    yield path or "/", scalars
    for f, k, v in nodes:
        if k == "msg":
            yield from iter_msgs(v, f"{path}/{f}" if path else str(f))

# ---------------------------------------------------------------- analysis

def floats_of(scalars):
    return [(f, v) for f, k, v in scalars if k in ("f32", "f64") and math.isfinite(v)]


def summarize(samples, f, n_max):
    stats = defaultdict(lambda: dict(count=0, frames=set(), example=None))
    quat, vec3 = defaultdict(list), defaultdict(list)
    for idx, (off, size, t) in enumerate(samples[:n_max]):
        f.seek(off)
        tree = decode(f.read(size))
        for path, scalars in iter_msgs(tree):
            s = stats[path]
            s["count"] += 1
            s["frames"].add(idx)
            if s["example"] is None:
                s["example"] = scalars
            fl = floats_of(scalars)
            if len(fl) == 4:
                quat[path].append(math.sqrt(sum(v * v for _, v in fl)))
            elif len(fl) == 3:
                vec3[path].append(math.sqrt(sum(v * v for _, v in fl)))
    return stats, quat, vec3


def fmt_val(kind, v):
    if kind in ("f32", "f64"):
        return f"{v:.6g}"
    if kind == "bytes":
        return f"<{len(v)} bytes>"
    if kind == "str":
        return repr(v[:30])
    return str(v)


def report(stats, quat, vec3, n_frames):
    print(f"\n=== Structure (over {n_frames} frames) ===")
    print(f"{'path':<22}{'occ./frame':>11}  scalar fields (example)")
    for path in sorted(stats, key=lambda p: [int(x) for x in p.split('/') if x.isdigit()]):
        s = stats[path]
        per_frame = s["count"] / max(1, len(s["frames"]))
        example = ", ".join(f"{f}:{k}={fmt_val(k, v)}" for f, k, v in (s["example"] or [])[:8])
        print(f"{path:<22}{per_frame:>11.1f}  {example}")

    print("\n=== Quaternion candidates (4 floats, norm ≈ 1) ===")
    found = False
    for path, norms in quat.items():
        mean = sum(norms) / len(norms)
        if all(abs(x - 1) < 0.02 for x in norms):
            found = True
            print(f"  {path}: {len(norms)} occ., mean norm {mean:.4f}")
    if not found:
        print("  none (the attitude may be stored as integers or packed bytes)")

    print("\n=== 3D vector candidates (gyro / accelerometer / magnetometer) ===")
    for path, norms in vec3.items():
        mean = sum(norms) / len(norms)
        sd = math.sqrt(sum((x - mean) ** 2 for x in norms) / len(norms))
        hint = ""
        if 0.8 < mean < 1.2:
            hint = "  <- accelerometer in g?"
        elif 8.5 < mean < 11:
            hint = "  <- accelerometer in m/s²?"
        print(f"  {path}: {len(norms)} occ., mean |v| {mean:.4g} (std {sd:.3g}){hint}")


def export(samples, f, target, out):
    rows, cols = [], set()
    for idx, (off, size, t) in enumerate(samples):
        f.seek(off)
        occ = 0
        for path, scalars in iter_msgs(decode(f.read(size))):
            if path == target:
                row = {"frame": idx, "frame_time_s": round(t, 6), "occ": occ}
                for fnum, kind, v in scalars:
                    if kind in ("int", "f32", "f64"):
                        row[f"f{fnum}"] = v
                        cols.add(fnum)
                rows.append(row)
                occ += 1
    header = ["frame", "frame_time_s", "occ"] + [f"f{c}" for c in sorted(cols)]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows, columns {header[3:]}")


def frame_scalars(tree):
    """{(path, field): value} for the FIRST occurrence of every numeric scalar of a frame."""
    out = {}
    for path, scalars in iter_msgs(tree):
        for fnum, kind, v in scalars:
            if kind in ("int", "f32", "f64") and (path, fnum) not in out:
                out[(path, fnum)] = v
    return out


def read_field(mp4, path, field, track="djmd"):
    """Time series (t in s since the start of the clip, values) of a scalar field,
    first occurrence per frame. Used by mp4_to_rosbag.py."""
    f, tracks = load_tracks(mp4)
    trk = next((t for t in tracks if t["format"] == track), None)
    if not trk:
        raise ValueError(f"no {track} track")
    times, values = [], []
    for off, size, t in trk["samples"]:
        f.seek(off)
        v = frame_scalars(decode(f.read(size))).get((path, field))
        if v is not None:
            times.append(t)
            values.append(float(v))
    return times, values


def scan(samples, f, csv_out):
    """Goes through the WHOLE clip, summarizes every scalar and flags depth /
    pressure candidates (plausible values, slow variation)."""
    series, times = {}, []
    for idx, (off, size, t) in enumerate(samples):
        f.seek(off)
        scalars = frame_scalars(decode(f.read(size)))
        times.append(t)
        for key, v in scalars.items():
            series.setdefault(key, {})[idx] = v
        if idx % 500 == 0:
            print(f"  {idx}/{len(samples)} frames", end="\r")
    print()
    n = len(samples)
    ranges = [("depth m", 0, 80), ("depth cm", 0, 8000), ("pressure Pa", 8e4, 9e5),
              ("pressure hPa/mbar", 800, 9000), ("pressure kPa", 80, 900), ("pressure bar", 0.8, 9)]
    counter = "counter / timestamp"
    rows = []
    for (path, fnum), by_frame in series.items():
        v = [float(x) for x in by_frame.values()]
        if len(v) < 0.5 * n or not all(math.isfinite(x) for x in v):
            continue
        lo, hi = min(v), max(v)
        span = hi - lo
        if span == 0:
            continue
        steps = sorted(abs(b - a) for a, b in zip(v, v[1:]))
        slow = steps[len(steps) // 2] < 0.01 * span  # slow variation (pressure sensor)
        if all(b > a for a, b in zip(v, v[1:])):  # strictly increasing: counter or timestamp
            rows.append((path, fnum, len(v), lo, hi, False, [counter]))
            continue
        kinds = [name for name, a, b in ranges if a <= lo and hi <= b]
        rows.append((path, fnum, len(v), lo, hi, slow, kinds))
    print("\n=== Scalars that vary during the clip ===")
    print(f"{'field':<16}{'n':>7}{'min':>14}{'max':>14}  slow  possible meaning")
    for path, fnum, count, lo, hi, slow, kinds in sorted(rows, key=lambda r: (not (r[5] and r[6]), r[0], int(r[1]))):
        flag = "<-" if slow and kinds and kinds[0] != counter else "  "
        print(f"{path + ':' + str(fnum):<16}{count:>7}{lo:>14.6g}{hi:>14.6g}  {'yes' if slow else 'no':<4}  "
              f"{', '.join(kinds) if kinds else '-'} {flag}")
    print("\n'<-' = depth/pressure candidate: plausible values AND slow variation.\n"
          "To confirm: a clip going down then up should give a U-shaped curve.")
    if csv_out:
        keys = [(r[0], r[1]) for r in rows]
        with open(csv_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_s"] + [f"{p}:{k}" for p, k in keys])
            for idx, t in enumerate(times):
                w.writerow([idx, round(t, 6)] + [series[k].get(idx, "") for k in keys])
        print(f"-> {csv_out} (one column per varying scalar, ready to plot)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mp4")
    ap.add_argument("--track", default="djmd", help="track format (default djmd; try dbgi)")
    ap.add_argument("-n", type=int, default=60, help="number of frames analysed for the summary")
    ap.add_argument("--export", metavar="PATH", help="sub-message to export, e.g. 3/2/10")
    ap.add_argument("-o", "--out", default="export.csv")
    ap.add_argument("--scan", action="store_true",
                    help="go through the whole clip and look for depth / pressure fields")
    ap.add_argument("--scan-csv", help="with --scan: write the series of all varying scalars")
    a = ap.parse_args()

    f, tracks = load_tracks(a.mp4)
    print("Tracks:")
    for t in tracks:
        print(f"  format={t['format']} handler={t['handler']} "
              f"timescale={t['timescale']} samples={len(t['samples'])}")
    trk = next((t for t in tracks if t["format"] == a.track), None)
    if not trk:
        sys.exit(f"No '{a.track}' track: was stabilization on, or the FOV not Wide, when recording?")

    if a.scan:
        scan(trk["samples"], f, a.scan_csv)
    elif a.export:
        export(trk["samples"], f, a.export, a.out)
    else:
        report(*summarize(trk["samples"], f, a.n), min(a.n, len(trk["samples"])))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Conversion d'un MP4 DJI Osmo Action 5 Pro en rosbag (ROS 1 ou ROS 2) :
  /action5/imu    sensor_msgs/Imu    gyro (rad/s), accéléro (m/s²), attitude (quaternion caméra)
  /action5/image  sensor_msgs/CompressedImage (JPEG, défaut) ou sensor_msgs/Image (--raw)
  /action5/pressure  sensor_msgs/FluidPressure (Pa)             } avec --depth-field
  /action5/depth     geometry_msgs/PointStamped, z = profondeur  } (m, positive vers le bas)
  /action5/temperature  sensor_msgs/Temperature (°C)            avec --temp-field

Action 5 Pro (dvtm_ac204) : la piste djmd ne contient que l'attitude à ~1 kHz
(3/3/2/3) et un accéléro par image (3/2/10, en g, ~30 Hz). L'IMU écrite est alors
reconstruite : gyro dérivé de l'attitude, accéléro interpolé, à --imu-rate Hz.
Baromètre : 3/2/13:3 = pression (hPa), 3/2/13:2 = température (centièmes de °C) ;
dans un caisson étanche, il mesure l'air du caisson, pas la profondeur.

Pas besoin d'installation ROS :
    pip install rosbags telemetry-parser opencv-python numpy

Exemples :
    python mp4_to_rosbag.py DJI_xxxx.MP4 -o bag_ros2                    # ROS 2, sqlite3
    python mp4_to_rosbag.py DJI_xxxx.MP4 -o bag_ros2 --storage mcap     # ROS 2, mcap
    python mp4_to_rosbag.py DJI_xxxx.MP4 -o clip.bag --ros 1            # ROS 1
    python mp4_to_rosbag.py DJI_xxxx.MP4 -o bag --raw --scale 0.5       # images brutes, demi-résolution
    python mp4_to_rosbag.py DJI_xxxx.MP4 -o bag --depth-field 3/2/13:3 --depth-unit hpa --p-atm auto \
        --temp-field 3/2/13:2                                          # + pression/profondeur, température

dji_imu_extract.py doit être dans le même dossier (on réutilise sa fonction collect).

Temps : stamp = t0 + temps relatif au début de la vidéo. Par défaut, t0 est la
date de création écrite dans le MP4 (boîte mvhd), interprétée en UTC. DJI y
écrit parfois l'heure locale : force la valeur avec --t0 (secondes epoch) si
besoin. Les timestamps IMU de telemetry-parser sont déjà alignés sur la vidéo
pour les caméras DJI.
"""
import argparse
import math
import shutil
import struct
import sys
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import telemetry_parser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dji_imu_extract import collect, walk  # noqa: E402

G = 9.80665

# ---------------------------------------------------------------- utilitaires

def mp4_creation_time(path):
    """Date de création (epoch s) lue dans moov/mvhd, ou None."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        end, pos = f.tell(), 0

        def boxes(start, stop):
            p = start
            while p + 8 <= stop:
                f.seek(p)
                size, typ = struct.unpack(">I4s", f.read(8))
                hdr = 8
                if size == 1:
                    size, hdr = struct.unpack(">Q", f.read(8))[0], 16
                elif size == 0:
                    size = stop - p
                if size < hdr:
                    return
                yield typ, p + hdr, p + size
                p += size

        for typ, s, e in boxes(0, end):
            if typ == b"moov":
                for t2, s2, _ in boxes(s, e):
                    if t2 == b"mvhd":
                        f.seek(s2)
                        v = f.read(1)[0]
                        f.read(3)
                        ct = struct.unpack(">Q" if v == 1 else ">I", f.read(8 if v == 1 else 4))[0]
                        return ct - 2082844800 if ct else None  # 1904 -> 1970
    return None


def slerp(q0, q1, a):
    dot = sum(x * y for x, y in zip(q0, q1))
    if dot < 0:
        q1, dot = [-x for x in q1], -dot
    if dot > 0.9995:
        q = [x + a * (y - x) for x, y in zip(q0, q1)]
    else:
        th = math.acos(dot)
        s0, s1 = math.sin((1 - a) * th) / math.sin(th), math.sin(a * th) / math.sin(th)
        q = [s0 * x + s1 * y for x, y in zip(q0, q1)]
    n = math.sqrt(sum(x * x for x in q))
    return [x / n for x in q]


class QuatInterp:
    """Interpolation SLERP des quaternions (t en secondes) aux instants IMU."""

    def __init__(self, rows):
        self.t = [r[0] for r in rows]
        self.q = [list(r[1:5]) for r in rows]

    def __call__(self, t):
        if not self.t:
            return None
        i = bisect_left(self.t, t)
        if i <= 0:
            return self.q[0]
        if i >= len(self.t):
            return self.q[-1]
        t0, t1 = self.t[i - 1], self.t[i]
        return slerp(self.q[i - 1], self.q[i], (t - t0) / (t1 - t0) if t1 > t0 else 0.0)


def time_scale(rows, ref_dur_s, mode):
    """Facteur pour passer en secondes le t d'une série (ms ou s), en choisissant
    l'unité qui rend sa durée cohérente avec celle de la vidéo."""
    if mode == "ms":
        return 1e-3
    if mode == "s":
        return 1.0
    span = max(r[0] for r in rows) - min(r[0] for r in rows)
    return 1e-3 if abs(span * 1e-3 - ref_dur_s) < abs(span - ref_dur_s) else 1.0


def gyro_from_quats(qt):
    """Vitesse angulaire (rad/s, repère corps) par différences finies des
    quaternions : w = 2 * vec(conj(q_k) ⊗ q_k+1) / dt, valable si q est
    l'orientation corps -> monde. Renvoie une liste alignée sur qt."""
    out = []
    for k in range(len(qt)):
        a, b = (k, k + 1) if k + 1 < len(qt) else (k - 1, k)
        if a < 0:
            out.append([0.0, 0.0, 0.0])
            continue
        (t0, w0, x0, y0, z0), (t1, w1, x1, y1, z1) = qt[a], qt[b]
        dt = t1 - t0
        if dt <= 0:
            out.append(out[-1] if out else [0.0, 0.0, 0.0])
            continue
        if w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1 < 0:  # même hémisphère
            w1, x1, y1, z1 = -w1, -x1, -y1, -z1
        # conj(q0) ⊗ q1
        dw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
        dx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        dy = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
        dz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
        nv = math.sqrt(dx * dx + dy * dy + dz * dz)
        ang = 2 * math.atan2(nv, dw)
        s = ang / (nv * dt) if nv > 1e-12 else 2 / dt
        out.append([dx * s, dy * s, dz * s])
    return out

def djmd_vector(mp4, spec, fields=(2, 3, 4)):
    """Vecteur 3D par image dans la piste djmd (ex. accéléro 3/2/10, champs 2-4).
    Renvoie (t en s depuis le début du clip, valeurs (N,3)) ou (None, None)."""
    from djmd_inspect import read_field
    path = "3/2/10" if spec == "auto" else spec
    try:
        cols = [read_field(mp4, path, f) for f in fields]
    except ValueError:
        return None, None
    if not cols[0][0] or any(len(c[0]) != len(cols[0][0]) for c in cols):
        if spec != "auto":
            print(f"  accéléro {path} : champs {fields} absents ou incomplets")
        return None, None
    return np.asarray(cols[0][0]), np.stack([np.asarray(c[1]) for c in cols], 1)


def djmd_scalar(mp4, spec):
    """Série d'un champ CHEMIN:CHAMP, sans les valeurs sentinelles (entiers non signés 'négatifs')."""
    from djmd_inspect import read_field
    path, _, fld = spec.rpartition(":")
    t, v = read_field(mp4, path, int(fld))
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.abs(v) < 1e12
    return t[ok], v[ok]


# ---------------------------------------------------------------- rosbag

class BagWriter:
    def __init__(self, out, ros, storage, raw):
        from rosbags.typesys import Stores, get_typestore
        self.ros = ros
        self.ts = get_typestore(Stores.ROS1_NOETIC if ros == 1 else Stores.ROS2_HUMBLE)
        T = self.ts.types
        self.Header, self.Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
        self.Imu, self.Quat, self.Vec3 = T["sensor_msgs/msg/Imu"], T["geometry_msgs/msg/Quaternion"], T["geometry_msgs/msg/Vector3"]
        self.Img = T["sensor_msgs/msg/Image"] if raw else T["sensor_msgs/msg/CompressedImage"]

        if ros == 1:
            from rosbags.rosbag1 import Writer
            self.w = Writer(Path(out))
        else:
            from rosbags.rosbag2 import Writer
            try:
                from rosbags.rosbag2 import StoragePlugin
                plugin = StoragePlugin.MCAP if storage == "mcap" else StoragePlugin.SQLITE3
                self.w = Writer(Path(out), version=8, storage_plugin=plugin)
            except (ImportError, TypeError):  # anciennes versions de rosbags
                self.w = Writer(Path(out))
        self.w.open()
        self.c_imu = self.w.add_connection("/action5/imu", self.Imu.__msgtype__, typestore=self.ts)
        self.c_img = self.w.add_connection("/action5/image", self.Img.__msgtype__, typestore=self.ts)
        self.seq = {"imu": 0, "img": 0}

    def header(self, t_ns, frame_id, kind):
        stamp = self.Time(sec=int(t_ns // 1_000_000_000), nanosec=int(t_ns % 1_000_000_000))
        if self.ros == 1:
            self.seq[kind] += 1
            return self.Header(seq=self.seq[kind], stamp=stamp, frame_id=frame_id)
        return self.Header(stamp=stamp, frame_id=frame_id)

    def write(self, conn, msg, t_ns):
        ser = self.ts.serialize_ros1 if self.ros == 1 else self.ts.serialize_cdr
        self.w.write(conn, t_ns, ser(msg, msg.__msgtype__))

    def imu(self, t_ns, gyro_rad, accl_ms2, q):
        # covariance à 0 = inconnue ; [0] = -1 = mesure absente (convention REP 145)
        ocov, gcov, acov = (np.zeros(9, dtype=np.float64) for _ in range(3))
        if q is None:
            q, ocov[0] = [1.0, 0.0, 0.0, 0.0], -1.0
        if gyro_rad is None:
            gyro_rad, gcov[0] = [0.0, 0.0, 0.0], -1.0
        if accl_ms2 is None:
            accl_ms2, acov[0] = [0.0, 0.0, 0.0], -1.0
        msg = self.Imu(
            header=self.header(t_ns, "action5_imu", "imu"),
            orientation=self.Quat(x=q[1], y=q[2], z=q[3], w=q[0]),
            orientation_covariance=ocov,
            angular_velocity=self.Vec3(x=gyro_rad[0], y=gyro_rad[1], z=gyro_rad[2]),
            angular_velocity_covariance=gcov,
            linear_acceleration=self.Vec3(x=accl_ms2[0], y=accl_ms2[1], z=accl_ms2[2]),
            linear_acceleration_covariance=acov,
        )
        self.write(self.c_imu, msg, t_ns)

    def enable_depth(self):
        T = self.ts.types
        self.Press, self.PtS = T["sensor_msgs/msg/FluidPressure"], T["geometry_msgs/msg/PointStamped"]
        self.Point = T["geometry_msgs/msg/Point"]
        self.c_press = self.w.add_connection("/action5/pressure", self.Press.__msgtype__, typestore=self.ts)
        self.c_depth = self.w.add_connection("/action5/depth", self.PtS.__msgtype__, typestore=self.ts)
        self.seq["press"] = self.seq["depth"] = 0

    def depth(self, t_ns, pressure_pa, depth_m):
        self.write(self.c_press, self.Press(header=self.header(t_ns, "action5_depth", "press"),
                                            fluid_pressure=float(pressure_pa), variance=0.0), t_ns)
        # profondeur en m, positive vers le bas, dans point.z
        self.write(self.c_depth, self.PtS(header=self.header(t_ns, "action5_depth", "depth"),
                                          point=self.Point(x=0.0, y=0.0, z=float(depth_m))), t_ns)

    def enable_temperature(self):
        self.Temp = self.ts.types["sensor_msgs/msg/Temperature"]
        self.c_temp = self.w.add_connection("/action5/temperature", self.Temp.__msgtype__, typestore=self.ts)
        self.seq["temp"] = 0

    def temperature(self, t_ns, celsius):
        self.write(self.c_temp, self.Temp(header=self.header(t_ns, "action5_depth", "temp"),
                                          temperature=float(celsius), variance=0.0), t_ns)

    def image(self, t_ns, frame_bgr, jpeg_q):
        h = self.header(t_ns, "action5_camera", "img")
        if self.Img.__msgtype__.endswith("CompressedImage"):
            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            msg = self.Img(header=h, format="jpeg", data=buf.reshape(-1).astype(np.uint8))
        else:
            hh, ww = frame_bgr.shape[:2]
            msg = self.Img(header=h, height=hh, width=ww, encoding="bgr8", is_bigendian=0,
                           step=ww * 3, data=np.ascontiguousarray(frame_bgr).reshape(-1))
        self.write(self.c_img, msg, t_ns)

    def close(self):
        self.w.close()

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mp4")
    ap.add_argument("-o", "--out", required=True, help="dossier (ROS 2) ou fichier .bag (ROS 1)")
    ap.add_argument("--ros", type=int, choices=(1, 2), default=2)
    ap.add_argument("--storage", choices=("sqlite3", "mcap"), default="sqlite3", help="ROS 2 uniquement")
    ap.add_argument("--raw", action="store_true", help="sensor_msgs/Image bgr8 au lieu de JPEG (très volumineux)")
    ap.add_argument("--scale", type=float, default=1.0, help="facteur de redimensionnement des images")
    ap.add_argument("--every", type=int, default=1, help="ne garder qu'une image sur N")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--t0", type=float, default=None, help="temps epoch (s) du début de la vidéo")
    ap.add_argument("--time-unit", choices=("auto", "ms", "s"), default="auto",
                    help="unité du t des séries brutes et des quaternions")
    ap.add_argument("--gyro-unit", choices=("auto", "deg", "rad"), default="auto",
                    help="unité du gyro brut (si normalized_imu est vide)")
    ap.add_argument("--overwrite", action="store_true", help="écrase le bag de sortie s'il existe")
    ap.add_argument("--depth-field", metavar="CHEMIN:CHAMP",
                    help="champ djmd de la profondeur ou de la pression (trouvé avec djmd_inspect.py --scan), ex. 3/7:2")
    ap.add_argument("--depth-unit", choices=("m", "cm", "mm", "pa", "hpa", "mbar", "kpa", "bar"), default="m",
                    help="unité du champ --depth-field")
    ap.add_argument("--p-atm", default="101325",
                    help="pression atmosphérique (Pa) pour convertir pression <-> profondeur ; "
                         "'auto' = pression minimale du clip (clip commencé en surface)")
    ap.add_argument("--water-density", type=float, default=1025.0, help="kg/m³ (1025 mer, 1000 eau douce)")
    ap.add_argument("--accel-field", default="auto", metavar="CHEMIN|auto|none",
                    help="sous-message djmd de l'accéléro par image (Action 4/5 Pro : 3/2/10, champs 2-4, en g) ; "
                         "'auto' = 3/2/10 s'il existe, utilisé seulement si telemetry-parser ne fournit pas d'accéléro")
    ap.add_argument("--imu-rate", type=float, default=200.0,
                    help="cadence (Hz) de l'IMU reconstruite quand le gyro est dérivé des quaternions")
    ap.add_argument("--temp-field", metavar="CHEMIN:CHAMP",
                    help="champ djmd de la température (Action 5 Pro : 3/2/13:2, en centièmes de °C)")
    ap.add_argument("--temp-scale", type=float, default=0.01, help="facteur vers des °C (défaut 0.01)")
    ap.add_argument("--no-derived-gyro", action="store_true",
                    help="ne pas dériver la vitesse angulaire des quaternions s'il n'y a pas de gyro")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        if not a.overwrite:
            sys.exit(f"{out} existe déjà : supprime-le ou relance avec --overwrite")
        shutil.rmtree(out) if out.is_dir() else out.unlink()

    cap = cv2.VideoCapture(a.mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_dur = nframes / fps if fps else 0.0
    print(f"Vidéo : {nframes} images à {fps:.3f} fps ({vid_dur:.2f} s)")

    # --- télémétrie : on recense tout ce qui est disponible
    tp = telemetry_parser.Parser(a.mp4)
    print(f"Caméra : {tp.camera} {tp.model}")
    norm = tp.normalized_imu() or []
    series, quats = collect(tp.telemetry())
    print(f"Disponible : normalized_imu={len(norm)}, gyro brut={len(series['Gyroscope'])}, "
          f"accéléro brut={len(series['Accelerometer'])}, quaternions={len(quats)}")

    def to_seconds(rows):
        """Remet le t d'une série en secondes, relatif au début de la vidéo."""
        if not rows:
            return rows
        k = time_scale(rows, vid_dur, a.time_unit)
        rows = [(r[0] * k, *r[1:]) for r in rows]
        if rows[0][0] > vid_dur:  # horloge de l'appareil et non du clip
            t_first = rows[0][0]
            rows = [(r[0] - t_first, *r[1:]) for r in rows]
        return rows

    quats = to_seconds(quats)
    qi = QuatInterp(quats) if quats else (lambda t: None)

    # samples : liste de (t_s, gyro_rad ou None, accl brut ou None)
    samples, source = [], ""
    norm = [s for s in norm if s.get("gyro") or s.get("accl")]
    if norm:
        source = "normalized_imu (repère caméra Gyroflow)"
        samples = [(s["timestamp_ms"] * 1e-3,
                    [math.radians(v) for v in s["gyro"]] if s.get("gyro") else None,
                    list(s["accl"]) if s.get("accl") else None) for s in norm]
    elif series["Gyroscope"] or series["Accelerometer"]:
        source = "séries brutes (repère capteur)"
        gyr = to_seconds(series["Gyroscope"])
        acc = to_seconds(series["Accelerometer"])
        gunit = a.gyro_unit
        if gunit == "auto" and gyr:
            mags = sorted(math.sqrt(x * x + y * y + z * z) for _, x, y, z in gyr)
            gunit = "deg" if mags[int(0.99 * (len(mags) - 1))] > 15 else "rad"
            print(f"  gyro brut supposé en {gunit}/s (force-le avec --gyro-unit)")
        gk = math.pi / 180 if gunit == "deg" else 1.0
        # même horloge pour gyro et accéléro ? on fusionne par timestamp identique
        accd = {round(t, 6): [x, y, z] for t, x, y, z in acc}
        used = set()
        for t, x, y, z in gyr:
            key = round(t, 6)
            samples.append((t, [x * gk, y * gk, z * gk], accd.get(key)))
            used.add(key)
        samples += [(t, None, [x, y, z]) for t, x, y, z in acc if round(t, 6) not in used]
    elif quats and not a.no_derived_gyro:
        # IMU reconstruite : quaternions rééchantillonnés à --imu-rate, gyro par
        # différences finies (moyenne naturelle sur 1/imu-rate), accéléro par image
        # interpolé linéairement. Le contenu au-delà de ~15 Hz est perdu sur l'accéléro.
        t_lo, t_hi = quats[0][0], quats[-1][0]
        grid = np.arange(t_lo, t_hi, 1.0 / a.imu_rate)
        qg = [(t, *qi(t)) for t in grid]
        gyro = gyro_from_quats(qg)
        acc_t, acc_v = djmd_vector(a.mp4, a.accel_field) if a.accel_field != "none" else (None, None)
        if acc_t is not None:
            acc_i = np.stack([np.interp(grid, acc_t, acc_v[:, k], left=np.nan, right=np.nan) for k in range(3)], 1)
            samples = [(t, g, None if np.isnan(av).any() else list(av)) for t, g, av in zip(grid, gyro, acc_i)]
            source = (f"quaternions -> gyro dérivé + accéléro djmd {a.accel_field if a.accel_field != 'auto' else '3/2/10'} "
                      f"({len(acc_t)} mesures, ~{len(acc_t) / max(acc_t[-1] - acc_t[0], 1e-9):.0f} Hz), "
                      f"rééchantillonnés à {a.imu_rate:g} Hz")
        else:
            samples = [(t, g, None) for t, g in zip(grid, gyro)]
            source = f"quaternions seuls -> gyro dérivé à {a.imu_rate:g} Hz, pas d'accéléro"
    elif quats:
        source = "quaternions seuls"
        samples = [(q[0], None, None) for q in quats]
    if not samples:
        sys.exit("Aucune donnée inertielle exploitable. Relance dji_imu_extract.py --dump-json "
                 "sur ce fichier et envoie-moi la liste des CSV produits.")
    samples.sort(key=lambda s: s[0])
    imu_t = [s[0] for s in samples]
    imu_dur = imu_t[-1] - imu_t[0]

    # accéléro : g ou m/s² ? (norme médiane)
    an = sorted(math.sqrt(sum(v * v for v in s[2])) for s in samples if s[2])
    acc_scale = G if an and an[len(an) // 2] < 3 else 1.0
    print(f"IMU : source = {source}")
    print(f"  {len(samples)} échantillons sur {imu_dur:.2f} s (~{len(samples) / max(imu_dur, 1e-9):.0f} Hz)"
          + (f", accéléro {'en g -> converti en m/s²' if acc_scale != 1 else 'en m/s²'}" if an else ""))
    print(f"Attitude : {len(quats)} quaternions" if quats else
          "Attitude : aucun quaternion, orientation_covariance[0] = -1")

    t0 = a.t0 if a.t0 is not None else (mp4_creation_time(a.mp4) or 0.0)
    t0_ns = int(round(t0 * 1e9))
    print(f"t0 = {t0:.3f} s epoch")

    # --- profondeur / pression
    depth_rows = []
    if a.depth_field:
        ts_d, vs = djmd_scalar(a.mp4, a.depth_field)
        if not len(vs):
            sys.exit(f"Champ {a.depth_field} introuvable dans la piste djmd")
        rho_g = a.water_density * G
        to_pa = {"pa": 1.0, "hpa": 100.0, "mbar": 100.0, "kpa": 1000.0, "bar": 1e5}
        to_m = {"m": 1.0, "cm": 0.01, "mm": 0.001}
        if a.depth_unit in to_pa:
            p_pa = vs * to_pa[a.depth_unit]
            p_atm = float(p_pa.min()) if a.p_atm == "auto" else float(a.p_atm)
            d_m = (p_pa - p_atm) / rho_g
        else:
            d_m = vs * to_m[a.depth_unit]
            p_atm = float(a.p_atm) if a.p_atm != "auto" else 101325.0
            p_pa = p_atm + d_m * rho_g
        depth_rows = list(zip(ts_d, p_pa, d_m))
        print(f"Profondeur : {len(depth_rows)} mesures, de {d_m.min():.2f} à {d_m.max():.2f} m "
              f"(P_atm = {p_atm:.0f} Pa, ρ = {a.water_density:g} kg/m³)")
    else:
        hits = sorted({" / ".join(pth) for pth, _ in walk(tp.telemetry())
                       if any(w in " ".join(pth).lower() for w in ("depth", "pressure", "baro"))})
        if hits:
            print("Champs de profondeur/pression vus par telemetry-parser (non utilisés) : " + "; ".join(hits[:5]))

    temp_rows = []
    if a.temp_field:
        tt, tv = djmd_scalar(a.mp4, a.temp_field)
        temp_rows = list(zip(tt, tv * a.temp_scale))
        if temp_rows:
            print(f"Température : {len(temp_rows)} mesures, {min(r[1] for r in temp_rows):.2f} "
                  f"à {max(r[1] for r in temp_rows):.2f} °C")

    bag = BagWriter(a.out, a.ros, a.storage, a.raw)
    if temp_rows:
        bag.enable_temperature()
        for t, c in temp_rows:
            bag.temperature(t0_ns + int(round(t * 1e9)), c)
    if depth_rows:
        bag.enable_depth()
        for t, p_pa, d_m in depth_rows:
            bag.depth(t0_ns + int(round(t * 1e9)), p_pa, d_m)

    j = n_img = 0
    i = 0

    def flush_imu(until_s):
        nonlocal j
        while j < len(samples) and imu_t[j] <= until_s:
            t, g, acc = samples[j]
            bag.imu(t0_ns + int(round(t * 1e9)), g,
                    [v * acc_scale for v in acc] if acc else None, qi(t))
            j += 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps  # DJI enregistre à fréquence d'images constante
            flush_imu(t)
            if i % a.every == 0:
                if a.scale != 1.0:
                    frame = cv2.resize(frame, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                bag.image(t0_ns + int(round(t * 1e9)), frame, a.jpeg_quality)
                n_img += 1
            i += 1
            if i % 300 == 0:
                print(f"  {i}/{nframes} images", end="\r")
        flush_imu(float("inf"))
    finally:
        cap.release()
        bag.close()
    print(f"\nTerminé : {n_img} images et {j} messages IMU -> {a.out}")


if __name__ == "__main__":
    main()

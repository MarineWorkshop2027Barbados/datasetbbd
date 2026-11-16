#!/usr/bin/env python3
"""
Extraction IMU (brute + normalisée) et attitude (quaternions) d'un MP4
DJI Osmo Action 5 Pro, via telemetry-parser (la librairie de Gyroflow).

    pip install telemetry-parser
    python dji_imu_extract.py DJI_20261116_0001.MP4 [-o sorties/] [--dump-json]

Rappel : la caméra n'enregistre ces données que si la stabilisation (EIS /
RockSteady) est désactivée et le FOV en Wide.

Sorties (dans -o, par défaut à côté du MP4) :
  <nom>_gyro_raw.csv, <nom>_accl_raw.csv   données capteur telles que décodées
  <nom>_imu_norm.csv                       IMU remise dans le repère caméra
                                           Gyroflow (gyro en deg/s)
  <nom>_attitude.csv                       quaternions (w,x,y,z) + Euler ZYX en degrés
  <nom>_telemetry.json                     (option --dump-json) tout le contenu
                                           décodé, pour inspection
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
    sys.exit("telemetry-parser manquant : pip install telemetry-parser")


def walk(obj, path=()):
    """Parcourt une structure imbriquée et renvoie (chemin, dict) pour chaque dict."""
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
    """Repère, dans la sortie de tp.telemetry(), les séries gyro / accéléro /
    magnéto (dicts {t,x,y,z}) et les quaternions ({t, v:{w,x,y,z}} ou {t,w,x,y,z}).
    Recherche par forme plutôt que par clé exacte, pour résister aux variations
    de version de telemetry-parser et de modèle de caméra."""
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

    def clean(rows):  # tri + suppression des doublons de timestamp
        out, seen = [], set()
        for r in sorted(rows, key=lambda r: r[0]):
            if r[0] not in seen:
                seen.add(r[0])
                out.append(r)
        return out

    return {k: clean(v) for k, v in series.items()}, clean(quats)


def quat_to_euler_zyx(w, x, y, z):
    """Quaternion de Hamilton -> (roll, pitch, yaw) en degrés, convention ZYX.
    Le repère et le sens de la rotation (corps->monde ou l'inverse) ne sont pas
    documentés par DJI : à valider sur un mouvement connu."""
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
    print(f"  {path.name}: {len(rows)} lignes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mp4")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--dump-json", action="store_true", help="écrit toute la télémétrie décodée en JSON")
    a = ap.parse_args()

    src = Path(a.mp4)
    out = Path(a.outdir) if a.outdir else src.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    tp = telemetry_parser.Parser(str(src))
    print(f"Caméra : {tp.camera}  Modèle : {tp.model}")

    telemetry = tp.telemetry()
    if a.dump_json:
        p = out / f"{stem}_telemetry.json"
        p.write_text(json.dumps(telemetry, default=str, indent=1))
        print(f"  {p.name} écrit")

    series, quats = collect(telemetry)

    # 1) Données brutes (unités et repère capteur telles que décodées ; le champ
    #    t est le timestamp fourni par le fichier, en ms en général : à vérifier
    #    sur la durée totale du clip)
    if series["Gyroscope"]:
        write_csv(out / f"{stem}_gyro_raw.csv", ["t", "gx", "gy", "gz"], series["Gyroscope"])
    if series["Accelerometer"]:
        write_csv(out / f"{stem}_accl_raw.csv", ["t", "ax", "ay", "az"], series["Accelerometer"])
    if series["Magnetometer"]:
        write_csv(out / f"{stem}_magn_raw.csv", ["t", "mx", "my", "mz"], series["Magnetometer"])

    # 2) IMU normalisée par telemetry-parser (repère caméra Gyroflow, gyro en deg/s,
    #    accéléro : vérifie la norme au repos, ~1 si en g, ~9.81 si en m/s²)
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

    # 3) Attitude : quaternions issus de la fusion faite par la caméra
    if quats:
        rows = [[t, w, x, y, z, *quat_to_euler_zyx(w, x, y, z)] for t, w, x, y, z in quats]
        write_csv(out / f"{stem}_attitude.csv",
                  ["t", "qw", "qx", "qy", "qz", "roll_deg", "pitch_deg", "yaw_deg"], rows)
    else:
        print("  Aucun quaternion trouvé : relance avec --dump-json pour inspecter, "
              "ou utilise djmd_inspect.py.")

    if not (series["Gyroscope"] or norm or quats):
        print("Aucune donnée IMU : vérifie que la vidéo a été tournée avec la "
              "stabilisation désactivée et le FOV en Wide.")


if __name__ == "__main__":
    main()

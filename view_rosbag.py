#!/usr/bin/env python3
"""
Lecture d'un bag créé par mp4_to_rosbag.py (ROS 1 ou ROS 2) : affiche les
images de /action5/image avec l'attitude de /action5/imu en surimpression
(roulis / tangage / lacet, horizon artificiel, gyro, accéléro).

    pip install rosbags opencv-python numpy          (+ matplotlib pour --plot)

    python view_rosbag.py bags/DJI_xxxx              # bag ROS 2 (dossier)
    python view_rosbag.py bags/DJI_xxxx.bag          # bag ROS 1
    python view_rosbag.py bags/DJI_xxxx --plot       # + courbes d'attitude et de gyro
    python view_rosbag.py bags/DJI_xxxx --save out.mp4 --no-show   # vidéo annotée, sans fenêtre

Touches : espace = pause, n = image suivante (en pause), + / - = vitesse,
          h = horizon on/off, q ou Échap = quitter
"""
import argparse
import math
import sys
import time
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

IMU_TOPIC = "/action5/imu"
IMG_TOPIC = "/action5/image"


def quat_to_euler_zyx(w, x, y, z):
    """Quaternion de Hamilton -> (roulis, tangage, lacet) en degrés, convention ZYX."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def decode_image(msg):
    if hasattr(msg, "format"):  # CompressedImage
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    buf = np.frombuffer(msg.data, np.uint8)
    enc = msg.encoding.lower()
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(buf.reshape(msg.height, msg.step), cv2.COLOR_GRAY2BGR)[:, :msg.width]
    img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img


# ---------------------------------------------------------------- lecture

def open_reader(bag):
    # les bags ROS 2 écrits par rosbags n'embarquent pas toujours les définitions
    # de messages : on fournit un typestore par défaut
    return AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE))


def load_imu(reader):
    conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
    if not conns:
        return None
    t, euler, quat, gyro, accl, has_q, has_g, has_a = [], [], [], [], [], [], [], []
    for conn, _, raw in reader.messages(connections=conns):
        m = reader.deserialize(raw, conn.msgtype)
        q = m.orientation
        t.append(stamp_ns(m))
        quat.append((q.w, q.x, q.y, q.z))
        euler.append(quat_to_euler_zyx(q.w, q.x, q.y, q.z))
        gyro.append((m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))
        accl.append((m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
        has_q.append(m.orientation_covariance[0] != -1)
        has_g.append(m.angular_velocity_covariance[0] != -1)
        has_a.append(m.linear_acceleration_covariance[0] != -1)
    order = np.argsort(t)
    pick = lambda v: np.asarray(v)[order]  # noqa: E731
    return dict(t=pick(t), euler=pick(euler), quat=pick(quat), gyro=pick(gyro), accl=pick(accl),
                has_q=pick(has_q), has_g=pick(has_g), has_a=pick(has_a))


def nearest(imu, t_ns):
    if imu is None or len(imu["t"]) == 0:
        return None
    i = bisect_left(imu["t"], t_ns)
    if i > 0 and (i == len(imu["t"]) or t_ns - imu["t"][i - 1] < imu["t"][i] - t_ns):
        i -= 1
    return i


# ---------------------------------------------------------------- affichage

def put(img, text, org, scale, color=(255, 255, 255)):
    # le fond semi-transparent du panneau suffit pour la lisibilité
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_horizon(img, roll_deg, pitch_deg, px_per_deg):
    """Horizon artificiel : ligne inclinée du roulis, décalée du tangage.
    Les signes dépendent du repère du quaternion DJI, à valider."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2 + pitch_deg * px_per_deg
    a = math.radians(-roll_deg)
    dx, dy = math.cos(a) * w, math.sin(a) * w
    cv2.line(img, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), (0, 255, 255), 2, cv2.LINE_AA)
    # réticule fixe au centre de l'image
    c = (w // 2, h // 2)
    cv2.line(img, (c[0] - 40, c[1]), (c[0] - 10, c[1]), (0, 0, 255), 2)
    cv2.line(img, (c[0] + 10, c[1]), (c[0] + 40, c[1]), (0, 0, 255), 2)
    cv2.circle(img, c, 3, (0, 0, 255), -1)


def overlay(img, t_ns, t_start_ns, imu, idx, show_horizon, speed, paused):
    h, w = img.shape[:2]
    s = max(0.45, w / 1600)
    lh = int(28 * s / 0.6)
    panel = img.copy()
    cv2.rectangle(panel, (0, 0), (int(430 * s / 0.6), int(lh * 7.6)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.45, img, 0.55, 0, img)

    y = lh
    put(img, f"t = {(t_ns - t_start_ns) * 1e-9:8.3f} s   x{speed:g}{'  PAUSE' if paused else ''}", (10, y), s)
    if idx is None:
        put(img, "pas d'IMU dans le bag", (10, y + lh), s, (0, 0, 255))
        return
    dt_ms = (int(imu["t"][idx]) - t_ns) * 1e-6
    y += lh
    put(img, f"IMU la plus proche : {dt_ms:+.1f} ms", (10, y), s, (200, 200, 200))
    y += lh
    if imu["has_q"][idx]:
        r, p, yw = imu["euler"][idx]
        put(img, f"roulis  {r:+8.2f} deg", (10, y), s, (0, 255, 255))
        put(img, f"tangage {p:+8.2f} deg", (10, y + lh), s, (0, 255, 255))
        put(img, f"lacet   {yw:+8.2f} deg", (10, y + 2 * lh), s, (0, 255, 255))
        if show_horizon:
            draw_horizon(img, r, p, px_per_deg=h / 90)
    else:
        put(img, "pas d'orientation", (10, y), s, (0, 0, 255))
    y += 3 * lh
    g = np.degrees(imu["gyro"][idx])
    put(img, ("gyro  " + " ".join(f"{v:+7.1f}" for v in g) + " deg/s") if imu["has_g"][idx]
        else "gyro : absent", (10, y), s)
    a = imu["accl"][idx]
    put(img, (f"accl  " + " ".join(f"{v:+6.2f}" for v in a) + f"  |a|={np.linalg.norm(a):.2f} m/s2")
        if imu["has_a"][idx] else "accl : absent", (10, y + lh), s)


def plot_imu(imu):
    import matplotlib.pyplot as plt
    t = (imu["t"] - imu["t"][0]) * 1e-9
    n = 3 if imu["has_a"].any() else 2
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=(11, 2.8 * n))
    for k, lab in enumerate(("roulis", "tangage", "lacet")):
        ax[0].plot(t[imu["has_q"]], imu["euler"][imu["has_q"], k], label=lab)
    ax[0].set_ylabel("attitude (deg)")
    for k, lab in enumerate("xyz"):
        ax[1].plot(t[imu["has_g"]], np.degrees(imu["gyro"][imu["has_g"], k]), label=f"ω{lab}")
    ax[1].set_ylabel("gyro (deg/s)")
    if n == 3:
        for k, lab in enumerate("xyz"):
            ax[2].plot(t[imu["has_a"]], imu["accl"][imu["has_a"], k], label=f"a{lab}")
        ax[2].set_ylabel("accl (m/s²)")
    for x in ax:
        x.legend(loc="upper right")
        x.grid(alpha=0.3)
    ax[-1].set_xlabel("temps (s)")
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="dossier d'un bag ROS 2 ou fichier .bag ROS 1")
    ap.add_argument("--width", type=int, default=1280, help="largeur d'affichage (px)")
    ap.add_argument("--speed", type=float, default=1.0, help="vitesse de lecture")
    ap.add_argument("--start", type=float, default=0.0, help="début de la lecture (s depuis le début du bag)")
    ap.add_argument("--plot", action="store_true", help="trace l'attitude et l'IMU (matplotlib)")
    ap.add_argument("--save", metavar="VIDEO.mp4", help="enregistre la vidéo annotée")
    ap.add_argument("--no-show", action="store_true", help="pas de fenêtre (avec --save)")
    a = ap.parse_args()

    with open_reader(a.bag) as reader:
        print("Topics :")
        for c in reader.connections:
            print(f"  {c.topic:<20} {c.msgtype:<32} {c.msgcount} messages")

        imu = load_imu(reader)
        if imu is not None and len(imu["t"]):
            dur = (imu["t"][-1] - imu["t"][0]) * 1e-9
            print(f"IMU : {len(imu['t'])} messages sur {dur:.2f} s "
                  f"(~{len(imu['t']) / max(dur, 1e-9):.0f} Hz), orientation "
                  f"{'présente' if imu['has_q'].any() else 'absente'}")
            if a.plot:
                plot_imu(imu)
        else:
            imu = None
            print(f"Pas de topic {IMU_TOPIC}")

        img_conns = [c for c in reader.connections if c.topic == IMG_TOPIC]
        if not img_conns:
            if a.plot:
                import matplotlib.pyplot as plt
                plt.show()
            sys.exit(f"Pas de topic {IMG_TOPIC}")

        t_start = reader.start_time
        t_from = t_start + int(a.start * 1e9)
        speed, paused, show_h = a.speed, False, True
        writer, win = None, "action5"
        if not a.no_show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        prev_stamp = prev_wall = None
        n = 0

        for conn, t_bag, raw in reader.messages(connections=img_conns, start=t_from):
            msg = reader.deserialize(raw, conn.msgtype)
            img = decode_image(msg)
            if img is None:
                continue
            if img.shape[1] != a.width:
                img = cv2.resize(img, (a.width, int(img.shape[0] * a.width / img.shape[1])),
                                 interpolation=cv2.INTER_AREA)
            t_ns = stamp_ns(msg)
            overlay(img, t_ns, t_start, imu, nearest(imu, t_ns), show_h, speed, paused)
            n += 1

            if a.save:
                if writer is None:
                    fps = 30.0
                    if conn.msgcount > 1 and reader.duration:
                        fps = conn.msgcount / (reader.duration * 1e-9)
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (img.shape[1], img.shape[0]))
                writer.write(img)
            if a.no_show:
                if n % 100 == 0:
                    print(f"  {n} images", end="\r")
                continue

            # cadence de lecture calée sur les timestamps
            now = time.monotonic()
            delay = 1
            if prev_stamp is not None and not paused:
                target = (t_ns - prev_stamp) * 1e-9 / speed
                delay = max(1, int((target - (now - prev_wall)) * 1000))
            cv2.imshow(win, img)
            key = cv2.waitKey(0 if paused else delay) & 0xFF
            prev_stamp, prev_wall = t_ns, time.monotonic()
            while True:
                if key in (ord("q"), 27):
                    raise SystemExit
                if key == ord(" "):
                    paused = not paused
                    if paused:
                        key = cv2.waitKey(0) & 0xFF
                        continue
                elif key in (ord("+"), ord("=")):
                    speed *= 2
                elif key == ord("-"):
                    speed /= 2
                elif key == ord("h"):
                    show_h = not show_h
                elif key == ord("n") and paused:
                    break
                if paused:  # en pause, on attend une touche
                    key = cv2.waitKey(0) & 0xFF
                    continue
                break

        if writer is not None:
            writer.release()
            print(f"\nVidéo annotée : {a.save}")
        if not a.no_show:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        cv2.destroyAllWindows()
        if e.code not in (None, 0):
            raise

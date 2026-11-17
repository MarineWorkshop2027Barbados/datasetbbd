# datasetbbd

This project includes scripts for data gathering during the McGill Barbados workshop 2027: a DJI Osmo Action 5 Pro mounted on a Deep Trekker ROV, with the camera data and the ROV log merged into a single synchronized ROS bag.

The pipeline:

1. **Configure the camera** so that it records its inertial data.
2. **Extract the camera data**: CSV (`dji_imu_extract.py`) and ROS bag (`mp4_to_rosbag.py`).
3. **Analyse the Deep Trekker log** (`rov_analysis.py`).
4. **Synchronize** the camera and the ROV (`sync_rov_camera.py`).
5. **Merge** everything into one bag on the ROV clock (`merge_bag.py`).
6. **Check** the result (`view_rosbag.py`), and inspect the raw camera metadata if needed (`djmd_inspect.py`).

The ROV log is the CSV written by the lab's `log_rov_csv.py` (one row per polling of the ROV state, with a `t_unix` column).

## Installation

### Requirements

- Python 3.8 or later
- `ffmpeg` / `ffprobe` (system package, only used to check the clips in section 1)
- no ROS installation needed, even for the bags

### Python libraries

| Library | Used by | Purpose |
|---|---|---|
| `telemetry-parser` | `dji_imu_extract.py`, `mp4_to_rosbag.py` | decodes the DJI `djmd` metadata track (Gyroflow) |
| `rosbags` | `mp4_to_rosbag.py`, `sync_rov_camera.py`, `merge_bag.py`, `view_rosbag.py` | writes and reads ROS 1 and ROS 2 bags without ROS |
| `opencv-python` | `mp4_to_rosbag.py`, `view_rosbag.py` | video decoding, JPEG encoding and display |
| `numpy` | all scripts but `djmd_inspect.py` | signal processing |
| `matplotlib` | `rov_analysis.py`, `sync_rov_camera.py`, `view_rosbag.py --plot` | figures |

`djmd_inspect.py` only uses the Python standard library.

### Setup

```bash
git clone https://github.com/MarineWorkshop2027Barbados/datasetbbd.git
cd datasetbbd

# virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# all the libraries at once
pip install -r requirements.txt
# or explicitly:
pip install telemetry-parser rosbags opencv-python numpy matplotlib

# ffprobe (to check the clips)
sudo apt install ffmpeg            # Ubuntu / Debian
brew install ffmpeg                # macOS
```

Check the installation:

```bash
python -c "import telemetry_parser, rosbags, cv2, numpy, matplotlib; print('OK')"
```

Keep all the scripts in the same folder: they import functions from each other.

## Quick start

For one dive, with the camera clip `DJI_xxxx.MP4` and the ROV log `rov_xxxx.csv`:

```bash
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx --temp-field 3/2/13:2      # camera bag
python rov_analysis.py rov_xxxx.csv -o reports/rov_xxxx                            # ROV report
python sync_rov_camera.py rov_xxxx.csv bags/DJI_xxxx -o sync/DJI_xxxx             # clock offset
python merge_bag.py bags/DJI_xxxx rov_xxxx.csv --sync sync/DJI_xxxx/sync.json -o merged/DJI_xxxx   # merged bag
python view_rosbag.py merged/DJI_xxxx                                              # check
```

Every script lists its options with `--help`.

---

## 1. Configure the DJI Osmo Action 5 Pro

The camera doesn't write its inertial data to a separate file. It stores them **inside the MP4**, in a timed metadata track called `djmd` (DJI protobuf), next to the video and audio tracks.

**The camera writes this track only under specific settings.** If the settings are wrong, the video looks normal but has no motion data, and it can't be fixed afterwards.

### Required settings

| Setting | Value | Why |
|---|---|---|
| Mode | **Video** (normal recording) | Slow motion, timelapse and hyperlapse modes have not been tested: check with a test clip before using them |
| Stabilization (EIS) | **Off**: no RockSteady, RockSteady+, HorizonBalancing or HorizonSteady | With EIS on, the camera does not record motion data |
| FOV | **Wide** | Other FOVs are cropped and corrected in camera, so no motion data is recorded |
| Firmware | Latest version (update via the DJI Mimo app) | Older firmware may not write the metadata track |

### Recommended settings

- **Constant frame rate**: choose a fixed resolution and frame rate (e.g. 2.7K 4:3, 30 fps) and keep it for the whole dataset.
- **Camera clock**: connect the camera to DJI Mimo before the session so that its clock syncs with the phone. The camera clock is corrected anyway by the synchronization (section 4), but a close starting point makes the result easier to check.
- **Field log**: write down the settings, site and dive of each clip.

### Synchronization manoeuvre

The camera and the ROV are synchronized by matching the rotations they both see (section 4). **At the start of each dive, with both recording, give the ROV a few sharp yaw turns** (and tilts if possible). A ROV holding heading and depth produces too little motion for a reliable synchronization.

### What the Action 5 Pro records

Identified with `djmd_inspect.py` (section 6) and the [ExifTool DJI tag tables](https://exiftool.sourceforge.net/TagNames/DJI.html) (`dvtm_ac204.proto`):

| `djmd` field | Content | Rate |
|---|---|---|
| `3/3/2/3` | attitude quaternion (w, x, y, z) from the camera's sensor fusion | ~1 kHz |
| `3/3/2:1` | timestamp of the attitude samples (µs) | ~1 kHz |
| `3/2/10:2,3,4` | accelerometer X, Y, Z (g) | one per frame (~30 Hz) |
| `3/2/9` | attitude quaternion | one per frame |
| `3/2/13:3` | barometric pressure (hPa) | one per frame |
| `3/2/13:2` | temperature (hundredths of °C, assumed) | one per frame |
| `3/2/6:1` | color temperature (K) | one per frame |
| `3/2/3:1` | ISO | one per frame |
| `3/1:1`, `3/1:2` | frame number, frame timestamp (µs) | one per frame |

**There is no raw gyroscope and no high-rate accelerometer** in the file: the gyroscope is derived from the 1 kHz attitude (section 2).

**In a waterproof housing**, the barometer measures the air sealed inside the housing, not the water pressure. The depth comes from the ROV (section 5).

### Check before going in the field

Record a short test clip and check that the metadata track is there:

```bash
ffprobe -hide_banner DJI_xxxx.MP4 2>&1 | grep -i "data"
```

You should see a data stream tagged `djmd` (and usually a `dbgi` one). If there is no such stream, check the stabilization and FOV settings and record again.

---

## 2. Extract the camera data

### To CSV: `dji_imu_extract.py`

Uses [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser), the library behind [Gyroflow](https://gyroflow.xyz).

```bash
python dji_imu_extract.py DJI_xxxx.MP4 -o output/ [--dump-json]
```

| File | Content |
|---|---|
| `<name>_attitude.csv` | `t, qw, qx, qy, qz, roll_deg, pitch_deg, yaw_deg`: camera attitude (Hamilton quaternion) and ZYX Euler angles |
| `<name>_gyro_raw.csv`, `<name>_accl_raw.csv`, `<name>_imu_norm.csv` | raw / normalized IMU, when telemetry-parser provides it (not on the Action 5 Pro) |
| `<name>_telemetry.json` | (with `--dump-json`) all the decoded telemetry |

On the Action 5 Pro only `_attitude.csv` is written. The per-frame accelerometer, pressure and temperature can be exported with `djmd_inspect.py --export` (section 6), and are included in the bag.

### To a ROS bag: `mp4_to_rosbag.py`

Uses [rosbags](https://gitlab.com/ternaris/rosbags), a pure Python library: **no ROS installation is needed**. Writes ROS 1 (`.bag`) or ROS 2 (sqlite3 or mcap) bags.

```bash
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx                        # ROS 2 (sqlite3)
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx --storage mcap         # ROS 2 (mcap)
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx.bag --ros 1            # ROS 1
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx --scale 0.5 --every 2  # lighter images
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx --temp-field 3/2/13:2 \
    --depth-field 3/2/13:3 --depth-unit hpa --p-atm auto                     # + camera barometer
```

| Topic | Type | frame_id | Content |
|---|---|---|---|
| `/action5/imu` | `sensor_msgs/Imu` | `action5_imu` | angular velocity (rad/s), linear acceleration (m/s²), orientation (camera quaternion) |
| `/action5/image` | `sensor_msgs/CompressedImage` (JPEG) or `sensor_msgs/Image` (`bgr8`, `--raw`) | `action5_camera` | video frames |
| `/action5/temperature` | `sensor_msgs/Temperature` | `action5_depth` | temperature (°C), with `--temp-field` |
| `/action5/pressure`, `/action5/depth` | `sensor_msgs/FluidPressure`, `geometry_msgs/PointStamped` | `action5_depth` | camera barometer, with `--depth-field` (meaningless in a sealed housing) |

Covariances are set to 0 ("unknown", REP 145). When a measurement is missing, the first element of its covariance is set to -1.

| Option | Default | Description |
|---|---|---|
| `--ros {1,2}` | `2` | ROS version of the bag |
| `--storage {sqlite3,mcap}` | `sqlite3` | ROS 2 storage plugin |
| `--overwrite` | off | overwrite the output bag if it already exists |
| `--raw` | off | `sensor_msgs/Image` instead of JPEG. Very large: about 25 MB per 4K frame |
| `--scale` | `1.0` | image resize factor |
| `--every N` | `1` | keep one frame out of N |
| `--jpeg-quality` | `90` | JPEG quality |
| `--t0` | MP4 creation time | absolute time (epoch, s) of the first video frame |
| `--imu-rate` | `200` | rate (Hz) of the rebuilt IMU |
| `--accel-field` | `auto` | `djmd` sub-message of the per-frame accelerometer (`3/2/10`); `none` to disable |
| `--temp-field`, `--temp-scale` | none, `0.01` | `djmd` temperature field and factor to °C |
| `--depth-field`, `--depth-unit` | none, `m` | `djmd` pressure or depth field and its unit |
| `--p-atm` | `101325` | atmospheric pressure (Pa); `auto` = minimum pressure of the clip |
| `--water-density` | `1025` | kg/m³ (1025 sea water, 1000 fresh water) |
| `--time-unit`, `--gyro-unit` | `auto` | units of the raw series, when telemetry-parser provides them |

**Rebuilt IMU (Action 5 Pro).** The attitude quaternions (~1 kHz) are resampled at `--imu-rate`. The angular velocity is computed from them by finite differences, and the per-frame accelerometer (~30 Hz, in g) is converted to m/s² and interpolated on the same grid. This is a **pseudo-IMU**: the gyroscope comes from an attitude already filtered by DJI, and the accelerometer loses any content above ~15 Hz. The line `IMU: source = …` in the output tells which source was used.

---

## 3. Analyse the Deep Trekker log: `rov_analysis.py`

```bash
python rov_analysis.py rov_xxxx.csv -o reports/rov_xxxx
```

The script reads the CSV and finds the columns by the end of their name, i.e. the JSON path of the ROV state (`depth`, `heading`, `velocity.forward`, `position.local.x`, `waterTemp`, …). It prints the mapping it used; force a column with `--map KEY=COLUMN`.

It reports:

- **Sampling**: number of rows, start / end (UTC), period, gaps.
- **Dives**: in-water segments (`inWater`, or depth > 0.3 m), duration, maximum depth.
- **Statistics** of every navigation variable, and the **validity** of the DVL altitude and velocity, and the USBL uncertainty.
- **Sync manoeuvres**: yaw turns faster than `--yaw-rate` (15 deg/s by default), usable by section 4.

Outputs:

| File | Content |
|---|---|
| `summary.json` | everything above |
| `overview.png` | depth / altitude, pitch / roll / camera tilt, heading, DVL velocity, water temperature / batteries; dives shaded, fast yaw turns in red |
| `track.png` | USBL local position, and DVL dead reckoning (body velocity integrated with the heading) |

Conventions assumed for the Deep Trekker state: heading in degrees clockwise from north; pitch / roll in degrees; DVL velocity in m/s in the body frame (forward, rightward, downward); depth and altitude in m.

---

## 4. Synchronize: `sync_rov_camera.py`

```bash
python sync_rov_camera.py rov_xxxx.csv bags/DJI_xxxx -o sync/DJI_xxxx
```

The camera is rigidly mounted on the ROV, so both see the same rotations. The angular speed |ω| does not depend on how the camera is mounted, so the clock offset is found by cross-correlating |ω| from the ROV attitude (heading / pitch / roll) with |ω| from the camera attitude. The search covers the whole ROV log, so any clock gap is found, including a time-zone error in the MP4.

The script then:

- measures the offset on successive windows to estimate the **clock drift** of the camera;
- estimates the **mounting rotation** ROV → camera IMU by aligning the relative rotations (hand-eye), which also settles the direction of the DJI quaternion.

Outputs:

| File | Content |
|---|---|
| `sync.json` | offset (`t_rov = t_cam + offset`), offset at the start of the clip, correlation quality, drift, mounting rotation; read by `merge_bag.py` |
| `sync.png` | |ω| of both after alignment, and the correlation curve (one sharp peak = reliable) |

The quality is `good` when the correlation peak is high and well above the next one. `UNCERTAIN` means the motion was not distinctive enough: look at `sync.png`, and do the sync manoeuvre on the next dives (section 1). The precision is limited by the ROV log rate (about 0.1 s at 2 Hz).

| Option | Default | Description |
|---|---|---|
| `--map KEY=COLUMN` | auto | force a ROV column (`t`, `heading`, `pitch`, `roll`) |
| `--rate` | `10` | analysis rate (Hz) |
| `--smooth` | `0.3` | |ω| smoothing window (s) |
| `--window` | `60` | window for the drift estimate (s) |
| `--delta` | `1.0` | interval of the relative rotations for the mounting rotation (s) |

---

## 5. Merge into one bag: `merge_bag.py`

```bash
python merge_bag.py bags/DJI_xxxx rov_xxxx.csv --sync sync/DJI_xxxx/sync.json -o merged/DJI_xxxx           # ROS 2 (mcap)
python merge_bag.py bags/DJI_xxxx rov_xxxx.csv --sync sync/DJI_xxxx/sync.json -o merged/DJI_xxxx.bag --ros 1
```

All the camera topics are copied with their stamps moved to the ROV clock: `t_rov = t_cam + offset_at_clip_start + drift × (t_cam − t0_cam)`. The ROV data are added as standard ROS messages:

| Topic | Type | frame_id | Content |
|---|---|---|---|
| `/rov/orientation` | `geometry_msgs/QuaternionStamped` | `rov_ned` | body FRD → local NED, from heading / pitch / roll |
| `/rov/attitude_deg` | `geometry_msgs/Vector3Stamped` | `rov_body_frd` | x = roll, y = pitch, z = heading (deg, as logged) |
| `/rov/depth` | `geometry_msgs/PointStamped` | `rov_body_frd` | z = depth (m, positive down) |
| `/rov/altitude` | `geometry_msgs/PointStamped` | `rov_body_frd` | z = DVL altitude above the seabed (m), valid samples only |
| `/rov/dvl/velocity` | `geometry_msgs/TwistStamped` | `rov_body_frd` | linear x / y / z = forward / rightward / downward (m/s), valid samples only |
| `/rov/usbl/position` | `geometry_msgs/PointStamped` | `rov_usbl_local` | x, y = USBL local position (as logged) |
| `/rov/water_temperature` | `sensor_msgs/Temperature` | `rov_body_frd` | water temperature (°C) |
| `/rov/raw/<column>` | `std_msgs/Float64` | | every other numeric CSV column (USBL uncertainty, batteries, motors, …), written when it changes; booleans as 0 / 1 |

| Option | Default | Description |
|---|---|---|
| `--ros {1,2}` | `2` | ROS version of the bag |
| `--storage {mcap,sqlite3}` | `mcap` | ROS 2 storage plugin |
| `--margin` | `10` | ROV data kept before / after the camera clip (s) |
| `--all-rov` | off | keep the whole ROV log |
| `--no-drift` | off | ignore the clock drift found by the sync |
| `--no-raw` | off | don't write the `/rov/raw/*` topics |
| `--map KEY=COLUMN` | auto | force a ROV column (see section 3) |
| `--overwrite` | off | overwrite the output bag if it exists |

---

## 6. Check and inspect

### View a bag: `view_rosbag.py`

Plays a camera bag or a merged bag back, with the camera attitude overlaid (roll, pitch, yaw, artificial horizon, gyroscope, accelerometer) and the depth (`/rov/depth` in a merged bag, otherwise `/action5/depth`).

```bash
python view_rosbag.py merged/DJI_xxxx                 # ROS 2 bag (folder)
python view_rosbag.py merged/DJI_xxxx.bag             # ROS 1 bag
python view_rosbag.py merged/DJI_xxxx --plot          # + attitude / gyro / accel plots
python view_rosbag.py merged/DJI_xxxx --save annotated.mp4 --no-show
```

| Option | Default | Description |
|---|---|---|
| `--width` | `1280` | display width (px) |
| `--speed` | `1.0` | playback speed |
| `--start` | `0` | start time (s from the beginning of the bag) |
| `--plot` | off | plot the attitude, gyroscope and accelerometer over time |
| `--save` | none | write the annotated video to a file |
| `--no-show` | off | no window (use with `--save`) |

Keys: `space` pause, `n` next frame (when paused), `+` / `-` speed, `h` horizon on/off, `q` or `Esc` quit.

### Inspect the raw camera metadata: `djmd_inspect.py`

Reads the `djmd` track directly, without telemetry-parser and without a protobuf schema. Use it to check a clip, find a field on another camera or firmware, or export fields that telemetry-parser doesn't decode.

```bash
python djmd_inspect.py DJI_xxxx.MP4                                  # structure, quaternion / vector candidates
python djmd_inspect.py DJI_xxxx.MP4 --export 3/2/10 -o accel.csv     # export a sub-message
python djmd_inspect.py DJI_xxxx.MP4 --scan --scan-csv scalars.csv    # varying scalars, depth / pressure candidates
```

Paths such as `3/2/10` are nested protobuf field numbers; a scalar field is written `PATH:FIELD` (e.g. `3/2/13:3`), as expected by the `--accel-field`, `--depth-field` and `--temp-field` options.

---

## Known limitations

- **Synchronization** relies on rotations: a dive without distinctive turns gives an uncertain offset. Its precision is limited by the ROV log rate.
- **Camera t0**: before synchronization, the MP4 creation time is read as UTC, but DJI sometimes writes local time; the synchronization corrects it.
- **Quaternion frame**: DJI doesn't document it; the synchronization estimates its direction and the mounting rotation, which should be checked against the physical mounting.
- **Rebuilt IMU**: the gyroscope is derived from a filtered attitude, and the accelerometer is sampled once per frame (~30 Hz) at an unknown instant within the frame (±16 ms).
- **ROV conventions**: heading, DVL axes and USBL local axes are taken as logged by the Deep Trekker; check them against a known manoeuvre.
- **Camera barometer**: inside a sealed housing it measures the housing air. The temperature field (`3/2/13:2`) is an assumption.

## Acknowledgements

The decoding of the DJI metadata relies on [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) by AdrianEddy (Gyroflow project). The identification of the `djmd` fields uses the [ExifTool DJI tag tables](https://exiftool.sourceforge.net/TagNames/DJI.html) by Phil Harvey.

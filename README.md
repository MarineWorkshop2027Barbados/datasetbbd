# datasetbbd

This project includes scripts for data gathering during the McGill Barbados workshop 2027.
It includes tutorials to:

1. configure a DJI Osmo Action 5 Pro so that it records its inertial data (IMU and attitude),
2. extract the IMU data and the camera attitude from the MP4 files to CSV,
3. convert a video and its inertial, pressure and temperature data into a ROS bag (ROS 1 or ROS 2),
4. play a bag back with the attitude overlaid on the images,
5. inspect the raw DJI metadata track.

## Installation

### Requirements

- Python 3.8 or later
- `ffmpeg` / `ffprobe` (system package, only used to check the clips in section 1)
- no ROS installation needed, even for the bags

### Python libraries

| Library | Used by | Purpose |
|---|---|---|
| `telemetry-parser` | `dji_imu_extract.py`, `mp4_to_rosbag.py` | decodes the DJI `djmd` metadata track (Gyroflow) |
| `rosbags` | `mp4_to_rosbag.py`, `view_rosbag.py` | writes and reads ROS 1 and ROS 2 bags without ROS |
| `opencv-python` | `mp4_to_rosbag.py`, `view_rosbag.py` | video decoding, JPEG encoding and display |
| `numpy` | `mp4_to_rosbag.py`, `view_rosbag.py` | image buffers, interpolation and covariance arrays |
| `matplotlib` | `view_rosbag.py` (`--plot` only) | attitude and IMU plots |

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

Keep all the scripts in the same folder: `mp4_to_rosbag.py` imports functions from `dji_imu_extract.py` and `djmd_inspect.py`.

## Usage overview

1. **Before the session**: configure the camera (section 1) and check a test clip.
2. **After the dive**: copy the `.MP4` files from the SD card (`DCIM/DJI_001/`) to your computer. The `.LRF` files are low-resolution proxies and are not needed.
3. **Extract to CSV** (section 2):
   ```bash
   python dji_imu_extract.py DJI_xxxx.MP4 -o output/
   ```
4. **Convert to a ROS bag** (section 3):
   ```bash
   python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx --temp-field 3/2/13:2           # ROS 2
   python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx.bag --ros 1 --temp-field 3/2/13:2   # ROS 1
   ```
5. **Check the bag**:
   ```bash
   ros2 bag info bags/DJI_xxxx       # ROS 2 (if ROS 2 is installed)
   rosbag info bags/DJI_xxxx.bag     # ROS 1 (if ROS 1 is installed)
   python view_rosbag.py bags/DJI_xxxx   # images + attitude (section 4)
   ```

To process a whole folder of clips:

```bash
mkdir -p output bags
for f in clips/*.MP4; do
  name=$(basename "$f" .MP4)
  python dji_imu_extract.py "$f" -o output/
  python mp4_to_rosbag.py "$f" -o "bags/$name" --overwrite --temp-field 3/2/13:2
done
```

Every script lists its options with `--help`:

```bash
python dji_imu_extract.py --help
python mp4_to_rosbag.py --help
python view_rosbag.py --help
python djmd_inspect.py --help
```

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

- **Camera clock**: connect the camera to DJI Mimo before the session so that its clock syncs with the phone. The ROS bag timestamps start from the creation time stored in the MP4. If you need accurate absolute time (e.g. syncing with other sensors), note the offset or use `--t0` (see section 3).
- **Constant frame rate**: choose a fixed resolution and frame rate (e.g. 2.7K 4:3, 30 fps) and keep it for the whole dataset.
- **Write down the settings** for each clip (resolution, fps, depth or site) in a field log.

### What the Action 5 Pro records

Identified with `djmd_inspect.py` (section 5) and the [ExifTool DJI tag tables](https://exiftool.sourceforge.net/TagNames/DJI.html) (`dvtm_ac204.proto`):

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

**There is no raw gyroscope and no high-rate accelerometer** in the file: the gyroscope is derived from the 1 kHz attitude (section 3).

**In a waterproof housing**, the barometer measures the air sealed inside the housing, not the water pressure: it can't be used as a depth sensor. Use a dive computer for depth.

### Check before going in the field

Record a short test clip and check that the metadata track is there:

```bash
ffprobe -hide_banner DJI_xxxx.MP4 2>&1 | grep -i "data"
```

You should see a data stream tagged `djmd` (and usually a `dbgi` one). If there is no such stream, check the stabilization and FOV settings and record again.

You can also run the extraction script (section 2) on the test clip: it should write a non-empty `_attitude.csv` file.

---

## 2. Extract IMU and attitude to CSV: `dji_imu_extract.py`

The script uses [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser), the library behind [Gyroflow](https://gyroflow.xyz), to decode the `djmd` track.

### Usage

```bash
python dji_imu_extract.py DJI_xxxx.MP4 -o output/ [--dump-json]
```

### Outputs

| File | Content |
|---|---|
| `<name>_gyro_raw.csv` | `t, gx, gy, gz`: raw gyroscope data, as decoded |
| `<name>_accl_raw.csv` | `t, ax, ay, az`: raw accelerometer data, as decoded |
| `<name>_imu_norm.csv` | `timestamp_ms, gx_dps, gy_dps, gz_dps, ax, ay, az, mx, my, mz`: IMU in the Gyroflow camera frame, gyro in deg/s |
| `<name>_attitude.csv` | `t, qw, qx, qy, qz, roll_deg, pitch_deg, yaw_deg`: camera attitude (Hamilton quaternion) and ZYX Euler angles |
| `<name>_telemetry.json` | (with `--dump-json`) all the decoded telemetry, for inspection |

A file is only written if the matching data exists in the clip. **On the Action 5 Pro, telemetry-parser only returns the attitude**, so only `_attitude.csv` is written. The per-frame accelerometer, pressure and temperature can be exported with `djmd_inspect.py --export` (section 5), and are included in the ROS bag (section 3).

---

## 3. Convert to a ROS bag: `mp4_to_rosbag.py`

The script writes the video frames, the IMU and, optionally, pressure and temperature into a bag. It uses [rosbags](https://gitlab.com/ternaris/rosbags), a pure Python library, so **no ROS installation is needed**. The same script writes ROS 1 (`.bag`) or ROS 2 (sqlite3 or mcap) bags.

### Usage

```bash
# ROS 2 (sqlite3)
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bag_ros2

# ROS 2 (mcap)
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bag_ros2 --storage mcap

# ROS 1
python mp4_to_rosbag.py DJI_xxxx.MP4 -o clip.bag --ros 1

# Raw images at half resolution, one frame out of two
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bag --raw --scale 0.5 --every 2

# With the barometer (pressure / depth) and the temperature
python mp4_to_rosbag.py DJI_xxxx.MP4 -o bag --overwrite \
    --depth-field 3/2/13:3 --depth-unit hpa --p-atm auto --temp-field 3/2/13:2
```

### Topics

| Topic | Type | frame_id | Content |
|---|---|---|---|
| `/action5/imu` | `sensor_msgs/Imu` | `action5_imu` | angular velocity (rad/s), linear acceleration (m/s²), orientation (camera quaternion, SLERP-interpolated at each IMU sample) |
| `/action5/image` | `sensor_msgs/CompressedImage` (JPEG, default) or `sensor_msgs/Image` (`bgr8`, with `--raw`) | `action5_camera` | video frames |
| `/action5/pressure` | `sensor_msgs/FluidPressure` | `action5_depth` | pressure (Pa), with `--depth-field` |
| `/action5/depth` | `geometry_msgs/PointStamped` | `action5_depth` | `point.z` = depth (m, positive down), with `--depth-field` |
| `/action5/temperature` | `sensor_msgs/Temperature` | `action5_depth` | temperature (°C), with `--temp-field` |

Covariances are set to 0, which means "unknown" (REP 145). When a measurement is missing, the first element of its covariance is set to -1.

### Options

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
| `--imu-rate` | `200` | rate (Hz) of the rebuilt IMU (see below) |
| `--accel-field` | `auto` | `djmd` sub-message of the per-frame accelerometer (`3/2/10` on the Action 4 / 5 Pro); `none` to disable |
| `--depth-field` | none | `djmd` field holding pressure or depth, e.g. `3/2/13:3` |
| `--depth-unit` | `m` | unit of `--depth-field`: `m`, `cm`, `mm`, `pa`, `hpa`, `mbar`, `kpa`, `bar` |
| `--p-atm` | `101325` | atmospheric pressure (Pa) for the pressure ↔ depth conversion; `auto` = minimum pressure of the clip |
| `--water-density` | `1025` | kg/m³ (1025 sea water, 1000 fresh water) |
| `--temp-field` | none | `djmd` field holding the temperature, e.g. `3/2/13:2` |
| `--temp-scale` | `0.01` | factor converting `--temp-field` to °C |
| `--time-unit {auto,ms,s}` | `auto` | time unit of the raw series and quaternions |
| `--gyro-unit {auto,deg,rad}` | `auto` | unit of the raw gyroscope |
| `--no-derived-gyro` | off | don't compute the angular velocity from the quaternions when no gyro is available |

### Where the IMU data come from

The script prints what it found in the clip, then uses the first source available:

1. the normalized IMU from telemetry-parser (Gyroflow camera frame);
2. the raw gyroscope and accelerometer series (sensor frame);
3. **the attitude quaternions (the Action 5 Pro case): the IMU is rebuilt.**
   - the quaternions (~1 kHz) are resampled on a `--imu-rate` grid (200 Hz by default);
   - the angular velocity is computed from them by finite differences;
   - the per-frame accelerometer (`3/2/10`, ~30 Hz, in g) is converted to m/s² and linearly interpolated on the same grid.

The line `IMU: source = …` in the output tells you which source was used.

The rebuilt IMU is a **pseudo-IMU**: the gyroscope comes from an attitude already filtered by DJI, and the accelerometer loses any content above ~15 Hz. It is fine for gravity direction and slow diver motions, but it doesn't follow the noise model of a raw IMU.

### Timestamps

`stamp = t0 + time since the start of the video`. Frame times are `index / fps`, which holds because the camera records at a constant frame rate. The IMU samples are already aligned with the video by the camera, so no extra synchronization is needed.

---

## 4. View a bag: `view_rosbag.py`

The script plays the bag back. It shows the images of `/action5/image` with the attitude from `/action5/imu` overlaid: roll, pitch, yaw, an artificial horizon, gyroscope and accelerometer values, and the depth when `/action5/depth` is present. It reads ROS 1 and ROS 2 bags, and doesn't need ROS either.

### Usage

```bash
python view_rosbag.py bags/DJI_xxxx                # ROS 2 bag (folder)
python view_rosbag.py bags/DJI_xxxx.bag            # ROS 1 bag
python view_rosbag.py bags/DJI_xxxx --plot         # + attitude / gyro / accel plots (matplotlib)
python view_rosbag.py bags/DJI_xxxx --save annotated.mp4 --no-show   # annotated video, no window
```

| Option | Default | Description |
|---|---|---|
| `--width` | `1280` | display width (px) |
| `--speed` | `1.0` | playback speed |
| `--start` | `0` | start time (s from the beginning of the bag) |
| `--plot` | off | plot the attitude, gyroscope and accelerometer over time |
| `--save` | none | write the annotated video to a file |
| `--no-show` | off | no window (use with `--save`, e.g. on a server) |

Keys: `space` pause, `n` next frame (when paused), `+` / `-` speed, `h` horizon on/off, `q` or `Esc` quit.

---

## 5. Inspect the raw metadata: `djmd_inspect.py`

The script reads the `djmd` track directly, without telemetry-parser and without a protobuf schema (Python standard library only). Use it to check what a clip contains, to find a field on another camera or firmware, or to export fields that telemetry-parser doesn't decode.

```bash
# structure: every sub-message, its number of occurrences per frame, example values,
# and candidate quaternions / 3D vectors (accelerometer in g or m/s²)
python djmd_inspect.py DJI_xxxx.MP4

# export a sub-message to CSV (one row per occurrence, one column per field)
python djmd_inspect.py DJI_xxxx.MP4 --export 3/2/10 -o accel.csv

# go through the whole clip and list the scalars that vary, with depth / pressure candidates
python djmd_inspect.py DJI_xxxx.MP4 --scan --scan-csv scalars.csv
```

Paths such as `3/2/10` are nested protobuf field numbers; a scalar field is written `PATH:FIELD` (e.g. `3/2/13:3`). These are the values expected by the `--accel-field`, `--depth-field` and `--temp-field` options of `mp4_to_rosbag.py`.

---

## Known limitations

- **t0 time zone**: the MP4 creation time is read as UTC, but DJI sometimes writes local time. Check it, and override it with `--t0` if needed.
- **Quaternion frame**: DJI doesn't document the frame or the direction of the quaternion (body→world or world→body). Validate it with a known rotation, e.g. 90° around one axis.
- **Heading**: it's not clear whether the camera uses a magnetometer in the fusion, so the yaw is probably relative and drifts.
- **Rebuilt IMU** (Action 5 Pro): the gyroscope is derived from a filtered attitude, and the accelerometer is sampled once per frame (~30 Hz); its exact sampling instant within the frame is unknown (±16 ms).
- **Barometer**: inside a sealed housing it measures the housing air, not the depth. The temperature field (`3/2/13:2`) is an assumption: check that it gives plausible values.
- **Raw units**: the time unit of the raw series (ms or s) and the gyroscope unit are guessed automatically. Check them on your first clip and force them with `--time-unit` and `--gyro-unit` if needed.

## Acknowledgements

The decoding of the DJI metadata relies on [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) by AdrianEddy (Gyroflow project). The identification of the `djmd` fields uses the [ExifTool DJI tag tables](https://exiftool.sourceforge.net/TagNames/DJI.html) by Phil Harvey.

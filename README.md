# datasetbbd

This project includes scripts for data gathering during the McGill Barbados workshop 2027.
It includes tutorials to:

1. configure a DJI Osmo Action 5 Pro so that it records its inertial data (IMU and attitude),
2. extract the IMU data and the camera attitude from the MP4 files to CSV,
3. convert a video and its IMU data into a ROS bag (ROS 1 or ROS 2).

## Installation

### Requirements

- Python 3.8 or later
- `ffmpeg` / `ffprobe` (system package, only used to check the clips in section 1)
- no ROS installation needed, even for the bags

### Python libraries

| Library | Used by | Purpose |
|---|---|---|
| `telemetry-parser` | both scripts | decodes the DJI `djmd` metadata track (Gyroflow) |
| `rosbags` | `mp4_to_rosbag.py` | writes ROS 1 and ROS 2 bags without ROS |
| `opencv-python` | `mp4_to_rosbag.py` | decodes the video (HEVC) and encodes JPEG frames |
| `numpy` | `mp4_to_rosbag.py` | image buffers and covariance arrays |

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
pip install telemetry-parser rosbags opencv-python numpy

# ffprobe (to check the clips)
sudo apt install ffmpeg            # Ubuntu / Debian
brew install ffmpeg                # macOS
```

Check the installation:

```bash
python -c "import telemetry_parser, rosbags, cv2, numpy; print('OK')"
```

## Usage overview

1. **Before the session**: configure the camera (section 1) and check a test clip.
2. **After the dive**: copy the `.MP4` files from the SD card (`DCIM/DJI_001/`) to your computer. The `.LRF` files are low-resolution proxies and are not needed.
3. **Extract to CSV** (section 2):
   ```bash
   python dji_imu_extract.py DJI_xxxx.MP4 -o output/
   ```
4. **Convert to a ROS bag** (section 3):
   ```bash
   python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx          # ROS 2
   python mp4_to_rosbag.py DJI_xxxx.MP4 -o bags/DJI_xxxx.bag --ros 1   # ROS 1
   ```
5. **Check the bag**:
   ```bash
   ros2 bag info bags/DJI_xxxx       # ROS 2 (if ROS 2 is installed)
   rosbag info bags/DJI_xxxx.bag     # ROS 1 (if ROS 1 is installed)
   ```

To process a whole folder of clips:

```bash
mkdir -p output bags
for f in clips/*.MP4; do
  name=$(basename "$f" .MP4)
  python dji_imu_extract.py "$f" -o output/
  python mp4_to_rosbag.py "$f" -o "bags/$name"
done
```

Every script lists its options with `--help`:

```bash
python dji_imu_extract.py --help
python mp4_to_rosbag.py --help
```

---

## 1. Configure the DJI Osmo Action 5 Pro

The camera doesn't write its inertial data to a separate file. It stores them **inside the MP4**, in a timed metadata track called `djmd` (DJI protobuf), next to the video and audio tracks. The track contains gyroscope and accelerometer samples, plus the attitude quaternion computed by the camera.

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
- **Constant frame rate**: choose a fixed resolution and frame rate (e.g. 4K 30 fps) and keep it for the whole dataset.
- **Write down the settings** for each clip (resolution, fps, depth or site) in a field log.

### Check before going in the field

Record a short test clip and check that the metadata track is there:

```bash
ffprobe -hide_banner DJI_xxxx.MP4 2>&1 | grep -i "data"
```

You should see a data stream tagged `djmd` (and usually a `dbgi` one). If there is no such stream, check the stabilization and FOV settings and record again.

You can also run the extraction script (section 2) on the test clip. It should produce non-empty CSV files.

---

## 2. Extract IMU and attitude to CSV: `dji_imu_extract.py`

The script uses [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser), the library behind [Gyroflow](https://gyroflow.xyz), to decode the `djmd` track.

### Installation

```bash
pip install telemetry-parser
```

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

A file is only written if the matching data exists in the clip.

---

## 3. Convert to a ROS bag: `mp4_to_rosbag.py`

The script writes the video frames and the IMU into a bag. It uses [rosbags](https://gitlab.com/ternaris/rosbags), a pure Python library, so **no ROS installation is needed**. The same script writes ROS 1 (`.bag`) or ROS 2 (sqlite3 or mcap) bags.

`mp4_to_rosbag.py` imports functions from `dji_imu_extract.py`: keep both files in the same folder.

### Installation

```bash
pip install rosbags telemetry-parser opencv-python numpy
```

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
```

### Topics

| Topic | Type | frame_id | Content |
|---|---|---|---|
| `/action5/imu` | `sensor_msgs/Imu` | `action5_imu` | angular velocity (rad/s), linear acceleration (m/s²), orientation (camera quaternion, SLERP-interpolated at each IMU sample) |
| `/action5/image` | `sensor_msgs/CompressedImage` (JPEG, default) or `sensor_msgs/Image` (`bgr8`, with `--raw`) | `action5_camera` | video frames |

Covariances are set to 0, which means "unknown" (REP 145). When a measurement is missing, the first element of its covariance is set to -1.

### Options

| Option | Default | Description |
|---|---|---|
| `--ros {1,2}` | `2` | ROS version of the bag |
| `--storage {sqlite3,mcap}` | `sqlite3` | ROS 2 storage plugin |
| `--raw` | off | `sensor_msgs/Image` instead of JPEG. Very large: about 25 MB per 4K frame |
| `--scale` | `1.0` | image resize factor |
| `--every N` | `1` | keep one frame out of N |
| `--jpeg-quality` | `90` | JPEG quality |
| `--t0` | MP4 creation time | absolute time (epoch, s) of the first video frame |
| `--time-unit {auto,ms,s}` | `auto` | time unit of the raw series and quaternions |
| `--gyro-unit {auto,deg,rad}` | `auto` | unit of the raw gyroscope |
| `--no-derived-gyro` | off | don't compute the angular velocity from the quaternions when no gyro is available |

### Where the IMU data come from

The script prints what it found in the clip, then uses the first source available:

1. the normalized IMU from telemetry-parser (Gyroflow camera frame);
2. the raw gyroscope and accelerometer series (sensor frame);
3. the quaternions only: the angular velocity is computed by finite differences and there is no acceleration.

The line `IMU : source = …` in the output tells you which source was used.

### Timestamps

`stamp = t0 + time since the start of the video`. Frame times are `index / fps`, which holds because the camera records at a constant frame rate. The IMU samples are already aligned with the video by the camera, so no extra synchronization is needed.

---

## Known limitations

- **t0 time zone**: the MP4 creation time is read as UTC, but DJI sometimes writes local time. Check it, and override it with `--t0` if needed.
- **Quaternion frame**: DJI doesn't document the frame or the direction of the quaternion (body→world or world→body). Validate it with a known rotation, e.g. 90° around one axis.
- **Heading**: it's not clear whether the camera uses its magnetometer in the fusion, so the yaw is probably relative and drifts.
- **Raw units**: the time unit of the raw series (ms or s) and the gyroscope unit are guessed automatically. Check them on your first clip and force them with `--time-unit` and `--gyro-unit` if needed.

## Acknowledgements

The decoding of the DJI metadata relies on [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) by AdrianEddy (Gyroflow project).


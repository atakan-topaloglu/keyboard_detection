# Project 4 Keyboard Vision To Robot Plan

## Decision

Use `atakan-code` as the perception, calibration, simulation, and evaluation sandbox first. Only after the detector-to-key-target pipeline is stable do we open a new branch in `arda-code` and wire those targets into SO-101 waypoint control.

The goal is not to train a large robot policy for typing. The goal is:

```text
keyboard image -> key detections -> keyboard pose/grid/PnP -> key target map -> robot waypoint presses
```

Imitation learning in `arda-code` already proves that the SO-101 can be commanded through LeRobot. The missing part is a reliable contract from vision to physical key targets.

## Repository Roles

### `atakan-code`

This repo owns the vision side.

Current files:

- `start_inference.sh`: starts the local Roboflow inference server.
- `run_keyboard_detection.sh`: runs the model on one image through the CLI.
- `render_keyboard_predictions.py`: Python client for the local inference server. It saves JSON predictions and labeled image overlays.
- `keyboard_pictures/`: sample keyboard images.

Planned responsibilities:

- Run keyboard/key detection on still images and camera frames.
- Normalize model output into a stable JSON schema.
- Fit a keyboard layout from detected keys.
- Run PnP or homography once camera intrinsics and key physical coordinates are known.
- Evaluate detection and geometry quality before touching the robot.
- Export target maps that `arda-code` can consume.

### `arda-code`

This repo owns physical robot control.

Current relevant files:

- `scripts/project4/rollout_progress_act.py`: current ACT rollout loop for the spacebar task.
- `src/lerobot/robots/so_follower/so_follower.py`: SO-101 observation/action API, including `robot.get_observation()` and `robot.send_action(...)`.
- `src/lerobot/robots/so_follower/robot_kinematic_processor.py`: existing end-effector safety and IK processor path.
- `move_to_position.py`: useful prototype for Cartesian-ish movement, but currently hard-coded and not production-ready.

Planned responsibilities:

- Create a new branch later for the robot typing controller.
- Read a key target map from `atakan-code` output.
- Move to safe pre-press waypoints.
- Use image-based correction near the key if the stylus is visible.
- Press with fixed calibrated depth and safety limits.

## Data Contract Between Repos

`atakan-code` should export one JSON file per frame or per calibration run. Proposed schema:

```json
{
  "image_path": "keyboard_pictures/example.jpg",
  "image_size": {"width": 640, "height": 480},
  "timestamp": "optional",
  "camera": {
    "intrinsics_id": "optional",
    "pose_source": "pnp|homography|manual|unknown"
  },
  "detections": [
    {
      "label": "a",
      "confidence": 0.91,
      "bbox_xywh": [320.0, 240.0, 34.0, 30.0],
      "center_px": [320.0, 240.0]
    }
  ],
  "keyboard_model": {
    "layout": "qwerty-us-lowercase-digits",
    "fit_score": 0.0,
    "method": "detected_keys_ransac|pnp|manual"
  },
  "key_targets": {
    "a": {
      "center_px": [320.0, 240.0],
      "center_keyboard_mm": [0.0, 0.0],
      "center_robot_m": null,
      "confidence": 0.91
    }
  }
}
```

The robot side should depend on this normalized target file, not on Roboflow-specific response details.

## Phase 1: Image Evaluation In `atakan-code`

Use your keyboard images first. This is the fastest way to answer whether the detector is good enough.

Tasks:

1. Add a Python wrapper around `render_keyboard_predictions.py` so detection can be called as a function, not only as a script.
2. Add `scripts/batch_detect_keyboards.py` to run all images in `keyboard_pictures/`.
3. Save normalized JSON to `outputs/predictions/`.
4. Save overlays to `outputs/overlays/`.
5. Add a summary report with:
   - number of detections per image
   - missing expected keys
   - duplicate key labels
   - confidence histogram
   - per-image latency
6. Create a small hand-labeled validation set from 5-10 images.

Success condition:

The overlay should put key centers close enough that we can visually trust the right key row and column. Exact robot accuracy does not need to be solved yet.

## Phase 2: Keyboard Grid And Pose Estimation

This is where Atakan's key detector becomes robot-useful.

Preferred approach:

1. Treat detections as noisy key-center observations.
2. Use known QWERTY layout geometry as the model.
3. Fit the detected keys to the template with RANSAC or robust least squares.
4. Reject detections that do not match the fitted grid.
5. Infer missing key centers from the fitted template.
6. If camera intrinsics and physical key spacing are available, run PnP.
7. Otherwise use homography as the first approximation.

Why this is easier than detecting keyboard corners:

- Corners are often hidden, rounded, reflective, or outside the crop.
- Individual key detections give many constraints.
- A QWERTY keyboard is a repeated grid with strong priors.
- Missing keys are acceptable if enough anchors are detected.

Outputs:

- `keyboard_model.fit_score`
- all `[0-9][a-z]` key centers in image pixels
- optional keyboard-frame coordinates in millimeters
- optional camera-to-keyboard transform from PnP

Success condition:

For each test image, the inferred centers for common keys like `a`, `s`, `d`, `f`, `j`, `k`, `l`, `space` should land near the real key centers in the overlay.

## Phase 3: Lightweight Simulation In `atakan-code`

We do not need a full MuJoCo sim first. Start with an image/geometry simulator.

Simulation levels:

### Level 0: Offline Image Replay

Run detection and grid fitting on saved images. Pretend the robot asks for a word and verify that each requested key maps to a visible target.

Example:

```text
input: image + word "hello"
output: h/e/l/l/o target centers overlaid on the image
```

### Level 1: Noisy Camera Geometry

Create synthetic perturbations:

- crop
- brightness changes
- blur
- rotation/perspective warp
- partial occlusion
- detector confidence threshold changes

Evaluate whether the fitted key map remains stable.

### Level 2: Virtual Servo Loop

Simulate stylus image position and target image position:

```text
target_key_px = key_targets["h"].center_px
stylus_px starts offset from target
controller outputs dx/dy correction
stylus_px moves closer with noise
stop when pixel error < threshold
```

This tests the control logic before touching the arm.

### Level 3: Robot Trace Export

Export a high-level action trace:

```json
[
  {"type": "move_above_key", "label": "h", "target_px": [100, 200]},
  {"type": "visual_servo_align", "label": "h"},
  {"type": "press", "label": "h"},
  {"type": "retract"}
]
```

This is the file `arda-code` can consume when we implement the physical branch.

## Phase 4: Real Camera Dry Run

Before robot motion, mount or hold the wrist camera in likely robot poses.

Tasks:

1. Capture overview images.
2. Capture near-key images with the gripper/stylus visible.
3. Run detection and grid fitting.
4. Detect stylus tip separately.
5. Overlay:
   - detected key boxes
   - fitted key centers
   - requested key target
   - stylus tip
   - pixel error vector

Success condition:

For a requested key, we can show a reliable vector from stylus tip to target key center in the camera image.

## Phase 5: New Branch In `arda-code`

Create this branch only after Phases 1-4 are stable enough:

```powershell
cd C:\git_repos\robot-learning\project\arda-code
git switch -c samuel-keyboard-typing
```

Initial files to add under `scripts/project4/`:

- `keyboard_typing_controller.py`
- `_keyboard_targets.py`
- `_so101_waypoints.py`

Implementation shape:

```text
load target map from atakan-code output
connect SO-101
move to safe overview pose
for each character:
    move to rough pre-press pose
    capture wrist image
    detect stylus tip and key target
    correct x/y with small closed-loop steps
    descend fixed calibrated depth
    retract
return home
```

Use the existing robot API:

- `robot.get_observation()` for joint state and camera frames.
- `robot.send_action({... ".pos": value ...})` for joint targets.
- Existing interpolation helpers as the model for safe gradual movement.

For Cartesian movement, prefer the existing LeRobot kinematics processors if they work on the robot PC. Keep the standalone `move_to_position.py` as a prototype reference, not the final integration.

## What Images You Can Provide

Useful images:

- overview of the full keyboard
- same keyboard from 2-3 wrist-camera heights
- close-up above `space`
- close-up above letter rows
- images with the gripper/stylus visible
- different lighting conditions
- one or two different keyboards if available

For each image, also record:

- whether it is from the wrist camera or another camera
- approximate camera pose: overview, above space, above letters
- keyboard model/layout if known
- whether the stylus tip is visible

## Key Risks

1. Roboflow dependency: first model fetch needs API/auth and network access.
2. Current scripts are Bash/Linux-style and may need WSL/Git Bash on Windows.
3. The detector output is boxes, not physical coordinates.
4. Keyboard corners are not reliable enough as the main method.
5. PnP needs camera intrinsics and physical key coordinates.
6. The robot branch must have safety limits before pressing real keys.
7. Pressing until motor stall should not be the normal contact strategy; use fixed depth plus compliance first.

## Immediate Next Steps

1. Add a normalized prediction exporter in `atakan-code`.
2. Add batch evaluation over `keyboard_pictures/`.
3. Add grid fitting from detected keys to QWERTY template.
4. Add overlay images showing inferred centers for every key.
5. Add a simple virtual servo simulation with stylus pixel position.
6. Capture real wrist-camera images with the stylus visible.
7. Only then branch `arda-code` and implement physical waypoint pressing.


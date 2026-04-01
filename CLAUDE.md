# NFL Player Tracking Pipeline
## Project Goal
Build a computer vision pipeline that extracts player tracking data from
NFL All-22 film (YouTube source, 720p). The extracted tracking data should
mimic NGS-style output: per-frame x/y field coordinates for all 22 players.
The primary use case is building tape-based athletic profiles for players
(speed, acceleration, turn radius, change-of-direction ability) by combining
CV-derived tracking with existing highlight play analysis.
## Accuracy Requirements
This project requires higher precision than basic route classification or
spacing analysis. The target metrics (acceleration, turn radius, COD) are
second-derivative quantities, meaning position errors compound through
differentiation. Specific requirements:
- Position accuracy: sub-yard RMSE target (ideally <0.5 yards)
- Temporal resolution: 30fps from YouTube source; may need trajectory
  smoothing (Kalman filter or spline interpolation) before computing
  velocity and acceleration derivatives
- Homography precision: must be tight enough that field mapping errors
  don't dominate the acceleration signal
- Validate derivatives (speed, accel) against NGS ground truth, not
  just raw position
## Available Data
- 3 NFL games (2019 season) downloaded as MP4 from YouTube at 720p
  - Wk 3: Ravens vs Chiefs (GameID: 2019092204)
  - Wk 8: Chiefs vs Packers (GameID: 2019102712)
  - Wk 10: Chiefs vs Titans (GameID: 2019111007)
- Each game has 2 camera angles: sideline wide, endzone tight
- NGS highlight tracking data for 7 plays across these games
  (in ~/Personal Research/ngs_highlights-master/play_data/)
  - Format: TSV files named SEASON_TEAM_GAMEID_PLAYID.tsv
  - Contains frame-level x/y coordinates at 10 Hz
## Camera Setup
- Sideline wide: good for lateral movement, east-west tracking
- Endzone tight: good for depth, north-south tracking
- No sideline tight or drone angles available for NFL
- Two-view fusion needed to resolve both axes of movement accurately
## Pipeline Architecture
Four stages, in order:
1. **Video acquisition** — Done. yt-dlp from YouTube.
2. **Player detection** — YOLOv12 via Ultralytics. Fine-tune on
   annotated All-22 frames. Detect all 22 players per frame.
3. **Player tracking** — BoT-SORT (preferred over ByteTrack for
   appearance-based re-ID). Need jersey color/number features for
   identity maintenance through crossing routes and brief occlusions.
4. **Field mapping** — Homography from yard lines/hash marks to
   real-world coordinates (100-yard field). Must be high-precision
   for derivative metrics. Use both camera angles to improve accuracy.
## Post-Processing
- Trajectory smoothing: Kalman filter or cubic spline on raw x/y
  positions before computing velocity/acceleration
- Velocity: first derivative of smoothed position
- Acceleration: second derivative, or first derivative of velocity
- Turn radius: computed from instantaneous curvature of trajectory
- Validate all derived metrics against NGS ground truth
## Key Design Decisions
- Each All-22 clip starts at the snap with players lined up. Identify
  all players at start of each play (jersey number recognition), then
  track through the play. No need to maintain identity across plays.
- Tracking confidence gating: when the tracker loses confidence in a
  player's identity (low re-ID score, occlusion, pile-up, player leaves
  frame on zoom-in, etc.), mark that player's trajectory as interrupted
  at the last confident position. Resume tracking when confidence
  recovers. Don't limit this to pile-ups — any situation where the
  tracker can't reliably maintain identity should trigger an interruption
  rather than output bad data.
- Two-camera fusion: combine sideline (strong east-west) and endzone
  (strong north-south) for best position estimates. Consider weighted
  averaging based on which view has less foreshortening for each player.
## Validation Approach
Compare CV-derived tracking against NGS ground truth for the 7 highlight
plays. Key metrics:
- Position RMSE (yards)
- Speed correlation and RMSE (yards/sec)
- Acceleration correlation and RMSE (yards/sec^2)
- Turn radius comparison on specific route breaks
## Tech Stack
- Python 3.10+
- ultralytics (YOLOv12)
- opencv-python
- numpy, pandas, scipy
- matplotlib/seaborn for visualization
- torch (GPU-accelerated inference)
## Commands
- Run tests: python -m pytest tests/ -v
- Run detection on a video: python src/detect.py --video <path> --output <path>
- Run full pipeline: python src/pipeline.py --video <path> --game-id <id>
## File Structure
- src/ — pipeline source code
- data/videos/ — raw game MP4s
- data/ngs/ — symlink to NGS highlight tracking data
- data/annotations/ — manually annotated training frames
- output/ — pipeline output (tracking CSVs, visualizations)
- tests/ — pytest test suite

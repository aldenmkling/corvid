# Can Computer Vision Replicate Player Tracking Data from All-22 Film?

## Abstract

The short answer is yes, with caveats. Computer vision pipelines combining modern object detectors (YOLOv8/v11), multi-object trackers (ByteTrack, DeepSORT), and homography-based field registration can extract player positions from broadcast or All-22 footage and project them onto a 2D field coordinate system. Commercial providers like SkillCorner already sell this as a product for American football, including NFL, UFL, FBS, and FCS games. Open-source implementations exist as well, though nearly all target soccer rather than American football. The technology works, but accuracy falls well short of the NFL's RFID-based Next Gen Stats system, particularly for speed and distance metrics. A 2025 validation study of three commercial CV tracking providers found position RMSE ranging from 1.68 to 16.39 meters and total distance bias as high as 24%, depending on camera feed and provider. For American football specifically, occlusion during pre-snap formations, huddles, and pile-ups at the line of scrimmage presents tracking challenges that are largely absent in soccer. The technology is usable for formation recognition, route classification, and relative positioning analysis. It is not yet a reliable substitute for sensor-based tracking when precise speed, acceleration, or sub-yard positioning matters.

## Context

The NFL's Next Gen Stats system, powered by Zebra Technologies RFID chips embedded in player shoulder pads, captures player coordinates at 10 Hz with accuracy on the order of inches. This data fuels the NFL's Big Data Bowl competition on Kaggle and underpins the league's public-facing analytics. Access to the raw tracking data outside of competition contexts is limited. For anyone wanting tracking-style data from college football, high school film, or historical NFL games, no sensor-based system exists. All-22 film, the wide-angle tactical camera showing all players on every snap, is the most natural candidate for CV-based extraction because it keeps all 22 players in frame throughout most plays.

The computer vision pipeline for this task has three stages: (1) detect players in each frame, (2) track detected players across frames to maintain identity, and (3) map pixel coordinates to real-world field coordinates via homography. Each stage has mature tooling, but each also introduces error that compounds through the pipeline.

## The Pipeline

**Detection.** Modern YOLO-family models (v8 through v11) achieve mAP50 scores above 0.85 on football player detection tasks with modest fine-tuning. Pre-trained models on Roboflow Universe can detect players, referees, and the ball out of the box. Detection accuracy degrades when players overlap (blocking at the line, gang tackles) or when the ball carrier is buried in a pile. American football uniforms, with bulky pads and complex jersey designs, make jersey number recognition harder than in basketball or soccer, complicating player identification.

**Tracking.** ByteTrack and BoT-SORT are the current standard multi-object trackers. They maintain player identity across frames using bounding box motion prediction and, optionally, appearance features. Identity switches (where the tracker confuses two players) are the primary failure mode, especially after occlusion events. In American football, the huddle is a guaranteed occlusion event on every play: 11 players cluster together, break, and must be re-identified as they move to their positions. Post-snap collisions at the line of scrimmage create another wave of occlusions. Soccer tracking pipelines rarely face this density of overlapping players.

**Homography.** Mapping pixel coordinates to field coordinates requires identifying known reference points (yard lines, hash marks, sideline intersections) and computing a perspective transformation. The All-22 camera is advantageous here because it is typically mounted high and roughly centered, producing less extreme perspective distortion than a broadcast sideline camera. Several open-source tools handle this: the Eagle project uses HRNet-based keypoint detection for pitch registration, and multiple GitHub repositories implement homography estimation from field markings. American football fields have dense, regular markings (yard lines every 5 yards, hash marks, numbers), which actually makes keypoint detection easier than on a soccer pitch.

## Commercial Solutions

SkillCorner is the most prominent commercial provider applying CV tracking to American football. They process All-22 camera feeds to generate player tracking data for the NFL, UFL, FBS, and FCS, delivering the output through TruMedia's front-end. Their system tracks all 22 players on all plays and produces physical performance metrics (speed, distance, acceleration) alongside contextual analytics. The UFL adopted SkillCorner's system league-wide for 2024 and expanded the partnership for 2025. Stats Perform (formerly Opta) and other providers offer similar CV-based tracking for soccer; their American football offerings are less publicly documented.

## Open-Source Options

Nearly all open-source football tracking repositories target soccer. The most complete pipelines include Eagle (YOLOv8 + HRNet keypoint detection + homography, converting broadcast footage to 2D tracking coordinates), JooZef315's football-tracking-data-from-TV-broadcast (YOLOv5 + DeepSORT + ResNet homography), and Roboflow's sports analysis toolkit (YOLOv8 + ByteTrack + supervision library). Adapting these to American football All-22 film requires retraining the detector on American football imagery (different player appearances, field markings, camera angles) and building a homography module calibrated to the American football field (100 yards plus end zones, hash marks at different widths than soccer). The detection retraining is straightforward with a few hundred annotated frames. The homography adaptation is more involved but tractable given the regular geometry of American football fields.

A Stanford CS231A project by Timothy Lee directly addressed player detection and tracking from NFL All-22 film, demonstrating that the approach is viable but noting challenges with player overlap and identity maintenance through formation changes.

## Accuracy Limitations

The most rigorous publicly available accuracy assessment is the Crang et al. (2025) study, which compared three commercial CV tracking providers against TRACAB Gen 5 (a multi-camera optical system used as ground truth) for a 2022 FIFA World Cup match. Position RMSE ranged from 1.68 to 16.39 meters across providers and camera feeds. Speed RMSE ranged from 0.34 to 2.38 m/s. Total match distance bias ranged from -21.8% to +24.3%. These numbers are for soccer, where occlusion is less severe than in American football; American football accuracy would likely be worse, particularly during and immediately after the snap.

For context, the NFL's RFID system operates at inch-level accuracy. CV-based tracking from a single camera is operating in a fundamentally different accuracy regime: useful for spatial patterns and relative positioning, but not for precise biomechanical or speed metrics.

## What You Can and Cannot Do

**Feasible with current tools:**

- Pre-snap formation recognition and alignment classification
- Route trees and route classification (especially for wide receivers and tight ends in space)
- Relative spacing and positioning analysis (gaps between defenders, cushion at the snap)
- General movement patterns and tendencies (where players end up, not precisely how fast they got there)
- Coverage shell identification and post-snap coverage classification

**Not reliably feasible from single-camera All-22:**

- Precise speed and acceleration measurements (error margins too large to distinguish meaningful differences between players)
- Exact distance traveled per play (bias can exceed 20%)
- Tracking through pile-ups, goal-line situations, and other severe occlusion events
- Reliable ball tracking (the football is small and frequently occluded)
- Sub-yard positioning accuracy needed for things like separation metrics

## Practical Path Forward

If you want to try this yourself, the most practical approach would be:

1. Start with a pre-trained YOLOv8 or YOLOv11 model and fine-tune on annotated All-22 frames. Roboflow makes this straightforward with their annotation and training pipeline.
2. Use ByteTrack or BoT-SORT for frame-to-frame tracking.
3. Implement homography using the yard lines and hash marks visible in All-22 film. The regularity of American football field markings is an advantage here.
4. Validate against a small set of plays where you have NFL Next Gen Stats data (available through Big Data Bowl datasets on Kaggle) to quantify your system's error.

The Big Data Bowl datasets on Kaggle provide real Next Gen Stats tracking data for specific game subsets, which could serve as ground truth for validating a CV pipeline's output on the same plays if you have the corresponding All-22 film.

## Conclusion

Computer vision can extract player tracking data from All-22 film, and commercial providers are already doing it at scale. The technology is mature enough for formation analysis, route classification, and spatial pattern recognition. It is not accurate enough to replicate the precision of the NFL's sensor-based system for speed, distance, or fine-grained positioning metrics. For a personal research project, the open-source toolchain (YOLO + ByteTrack + homography) is viable but will require retraining on American football imagery and careful validation. The biggest challenge specific to American football is maintaining player identity through huddles, pre-snap motion, and line-of-scrimmage collisions, where occlusion density exceeds what soccer-focused pipelines are designed to handle.

### Data Sources

- Crang, Z.L. et al. (2025). "Concurrent validity of computer-vision artificial intelligence player tracking software using broadcast footage." [arXiv:2508.19477](https://arxiv.org/abs/2508.19477)
- SkillCorner American Football product page: [skillcorner.com/sports/american-football](https://skillcorner.com/sports/american-football)
- UFL-SkillCorner partnership announcement: [skillcorner.com/articles/ufl-chooses-skillcorner](https://skillcorner.com/articles/ufl-chooses-skillcorner)
- Eagle: broadcast-to-tracking CV pipeline: [github.com/nreHieW/Eagle](https://github.com/nreHieW/Eagle)
- JooZef315 football tracking from TV broadcast: [github.com/JooZef315/football-tracking-data-from-TV-broadcast](https://github.com/JooZef315/football-tracking-data-from-TV-broadcast)
- Roboflow football player tracking tutorial: [blog.roboflow.com/track-football-players](https://blog.roboflow.com/track-football-players)
- NFL Next Gen Stats overview: [operations.nfl.com/gameday/technology/nfl-next-gen-stats](https://operations.nfl.com/gameday/technology/nfl-next-gen-stats/)
- NFL Big Data Bowl (Kaggle tracking data): [kaggle.com/competitions/nfl-big-data-bowl-2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025)
- Lee, T. "Using Computer Vision to Analyze All-22 NFL Film." Stanford CS231A project.
- Zheng, F. et al. (2025). "A review of computer vision technology for football videos." [mdpi.com/2078-2489/16/5/355](https://www.mdpi.com/2078-2489/16/5/355)
- Amazon Science: "A decade of NFL Next Gen Stats innovation." [amazon.science/blog/a-decade-of-nfl-next-gen-stats-innovation](https://www.amazon.science/blog/a-decade-of-nfl-next-gen-stats-innovation)

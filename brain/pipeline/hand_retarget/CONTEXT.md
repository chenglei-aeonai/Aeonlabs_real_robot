# Hand retargeting

Turns a human demonstration (a video of a hand doing a task) into motion of a UR5e arm carrying a Tesollo DG5F right hand, and replays it so the robot reproduces the task.

## Language

**Hand pose estimator**:
The AI model that extracts 3D hand keypoints from a video.
_Avoid_: Policy (reserved for learned controllers), tracker, detector

**Skeleton**:
The 21 hand keypoints (MANO joint set) of the right hand in one video frame, as output by the hand pose estimator.
_Avoid_: Pose (bare), landmarks, joints (those are the robot's)

**Human demonstration**:
One video of a hand doing a task, together with its sequence of skeletons expressed in the demonstration frame.
_Avoid_: Demonstration (bare, that is the robot demonstration), clip, recording

**Demonstration frame**:
The fixed, z-up frame a human demonstration is expressed in, the same for every video frame: origin at the right wrist in the first frame, x forward (the camera's viewing direction projected level), y left, z up.
_Avoid_: World frame (bare), camera frame (that is the unlevelled frame the estimator predicts in), paper frame (a later, measured replacement)

**Table plane**:
The horizontal plane where the table surface lies. By default it is the real table's height in simtoolreal (3.5 cm below the robot base), and it can be moved by hand, for the whole demonstration. The skeletons are never moved to respect it; if the hand goes below it, the table or the demonstration placement is moved instead.
_Avoid_: Ground, floor, table height (bare)

**Demonstration placement**:
The single rigid transform (position and yaw, pitch, roll) that puts a human demonstration's frame onto the robot's table, applied identically to every frame. It and the table plane's height are the only things that may be adjusted; the skeletons themselves are never modified.
_Avoid_: Offset, calibration, alignment

**Robot demonstration**:
A time series of robot joint angles (arm and hand) that the robot can execute, in the format simtoolreal_newton uses for its demonstration.
_Avoid_: Trajectory (bare), motion file

**Hand retargeting**:
Turning a skeleton into robot joint angles so the robot hand reproduces what the human hand does.
_Avoid_: Retargeting (bare, in simtoolreal_newton that means moving a robot demonstration to a new object pose), mapping

**Retargeter**:
One method of hand retargeting, called once per skeleton, which may keep state from the previous frame.
_Avoid_: Solver, mapper

**Retargeting method**:
Which retargeter is in use: fingertip IK (match robot fingertip positions to the human ones), direct joint mapping (convert human joint angles to robot joint angles, following the MANUS glove teleoperation), or vector retargeting (match the vectors from palm to fingertips and from thumb to fingertips, snapping near-touching pairs into contact).
_Avoid_: Strategy, mode

**Hand anchor**:
The centre of the knuckle row (the MCP joints of the long fingers), the point where the robot hand is placed to coincide with the human hand. Chosen over the wrist because the two hands have different palm lengths, and grasping happens at the fingers.
_Avoid_: Wrist target, palm origin, hand origin

**Hand scaling**:
Resizing the human skeleton to the robot hand's size before hand retargeting, either by one global factor or by a separate factor per finger.
_Avoid_: Normalisation, resizing (bare)

**Replay**:
Showing a human demonstration's retargeted motion on the robot in simulation, by setting joint angles directly without physics (kinematic replay).
_Avoid_: Playback, rollout (that is a policy acting), simulation (bare)

**Floating hand**:
The DG5F without the arm, whose palm is placed directly at the target pose; used as a debug mode that separates hand retargeting errors from arm reach problems.
_Avoid_: Free hand, detached hand

**Finger penetration**:
How deep the collision meshes of two different robot fingers go into each other. Touching is allowed (a pinch, fingers side by side in a fist); penetration is not.
_Avoid_: Collision (bare), clearance, contact (that is touching without penetrating)

**Fingertip error**:
The distance between each robot fingertip and the corresponding human fingertip, once both are expressed in the same frame and scale; the v1 measure of retargeting quality.
_Avoid_: Tracking error, accuracy

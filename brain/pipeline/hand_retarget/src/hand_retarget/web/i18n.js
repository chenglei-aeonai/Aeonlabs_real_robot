"use strict";
// UI copy only. Instructions, filenames, model IDs and diagnostic details retain their original content.
const messages = {
  "futureResult": ["Generated results", "生成结果"],
  "futureRGBVideo": ["Generated video", "生成视频"],
  "diffusionTitle": ["Future video + hand 3D", "未来视频和手部三维轨迹"],
  "serverPanelTitle": ["Server startup & checkpoint", "服务器启动和检查点选择"],
  "activeCheckpoint": ["Active checkpoint", "已加载检查点"],
  "serverJob": ["GPU job", "GPU 任务"],
  "serverNode": ["Node", "计算节点"],
  "aeonURL": ["aeon-webui URL / SSH tunnel", "aeon-webui 地址 / SSH 隧道"],
  "connectDiffusion": ["Connect / refresh", "连接 / 刷新"],
  "serverConnected": ["Connected", "已连接"],
  "gpuStopped": ["GPU server stopped", "GPU 服务未启动"],
  "serverConnectedNext": ["Connected to aeon-webui. Select a checkpoint and click Start server / load checkpoint to request the CSCS GPU allocation.", "已连接 aeon-webui。请选择检查点并点击启动服务器 / 加载检查点以申请 CSCS GPU 资源。"],
  "checkpointProfile": ["Checkpoint profile", "检查点配置"],
  "checkpointQueue": ["GPU queue", "GPU 队列"],
  "loadCheckpoint": ["Start server / load checkpoint", "启动服务器 / 加载检查点"],
  "stopCheckpoint": ["Stop server", "停止服务"],
  "diffusionSource": ["Converted conditioning frame", "转换后的条件图像"],
  "convertedRGB": ["Converted RGB", "转换后的 RGB 图像"],
  "capturedConditioning": ["Captured MANO GT + crop intrinsics", "采集的 MANO GT 和裁剪后内参"],
  "prepareConditioning": ["Export GT conditioning bundle", "导出 GT 条件数据"],
  "downloadConditioning": ["Download conditioning NPZ", "下载条件 NPZ"],
  "futureInstruction": ["What should the hand do next?", "接下来手应该做什么？"],
  "zeroContact": ["For contact-conditioned checkpoints: assume no initial contact or force", "接触条件模型：假设初始无接触、无力"],
  "diffusionHint": ["The model receives the converted image, measured intrinsics, and captured camera-space GT. It does not re-estimate the initial pose. Loading a checkpoint starts a GPU job on the connected inference service.", "模型接收转换后的图像、实测相机内参和相机坐标系 GT，不重新估计初始姿态。加载检查点将在所连接的推理服务上启动 GPU 任务。"],
  "generateFuture": ["Generate future video + hand 3D", "生成未来视频和手部三维轨迹"],
  "predictedHandVideo": ["Generated video with predicted hand skeleton", "生成视频和预测手部骨架"],
  "downloadFutureHands": ["Future hand 3D NPZ", "未来手部三维轨迹 NPZ"],
  "downloadConditioningInfo": ["Conditioning metadata", "条件数据说明"],
  "captureRobot": ["Capture robot camera", "采集机器人相机首帧"],
  "calibratedCameraHint": ["Camera intrinsics come from the applied capture/crop. Camera orientation comes from the saved calibration.", "相机内参来自已应用的采集与裁剪，相机方向来自保存的标定结果。"],
  "cropTitle": ["Frame the hand and table", "调整手和桌面的取景区域"],
  "cropWidth": ["Output width", "输出宽度"],
  "cropHeight": ["Output height", "输出高度"],
  "cropZoom": ["Crop zoom", "裁剪缩放"],
  "cropX": ["Horizontal position", "水平位置"],
  "cropY": ["Vertical position", "垂直位置"],
  "applyCrop": ["Apply crop", "应用裁剪"],
  "resetCrop": ["Reset hand/table framing", "重置手和桌面取景"],
  "cropHint": ["Apply saves width, height, zoom, and position as defaults for future captures in this browser. Intrinsics and projections follow the crop; 3D GT stays unchanged.", "应用后保存宽度、高度、缩放及位置，作为此浏览器后续采集的默认设置。内参和投影随裁剪更新，三维 GT 保持不变。"],
  "convertedPose": ["Captured GT MANO pose on converted image", "转换图像上的采集 GT MANO 关键点"],
  "captureRobotHint": ["Keep the arm and fingers stationary. Capture uses the robot PC's RealSense and measured joints; no motion is commanded.", "请保持机械臂和手指静止。使用机器人电脑连接的 RealSense 和实测关节状态，不发送运动指令。"],
  "robotOverlay": ["Robot joints", "机器人关节叠加图"],
  "manoOverlay": ["Mapped MANO joints", "映射后的 MANO 关键点"],
  "captureJSON": ["Camera intrinsics + MANO JSON", "相机内参和 MANO 关键点 JSON"],
  "captureImage": ["Undistorted image", "去畸变图像"],
  "convertHand": ["Convert robot hand and arm to human", "将机械手和机械臂转换成人手和手臂"],
  "pageTitle": ["Robot Demonstration Studio · Aeon Labs", "机器人示范工作台 · Aeon Labs"],
  "studio": ["ROBOT DEMONSTRATION STUDIO", "机器人示范工作台"],
  "robotics": ["ROBOTICS", "机器人"],
  "demonstrations": ["Demo", "动作示范"],
  "rightHand": ["Tesollo DG5F · Right", "Tesollo DG5F · 右手"],
  "robotFooter": ["UR5e + Tesollo DG5F · right hand", "UR5e + Tesollo DG5F · 右手"],
  "headline": ["From observation to action.", "从初始观察，到机器人动作。"],
  "intro": ["Upload a robot-view image, describe the task, and export a dexterous-hand trajectory.", "上传机器人视角，描述任务，生成可导出的灵巧手动作轨迹。"],
  "durationBadge": ["4-second demonstration", "4 秒示范"],
  "jointsBadge": ["60 Hz · 6 + 20 joints", "60 Hz · 6 + 20 个关节"],
  "inputTitle": ["Initial observation & task", "初始观察与任务"],
  "dropImage": ["Drop a first-person robot-hand image", "拖入机器手的第一视角图片"],
  "formats": ["PNG, JPEG or WebP · up to 16 MP", "PNG、JPEG 或 WebP · 最大 16 MP"],
  "inputCanvas": ["Input image; paint the hand region in manual mode", "输入图片；手动模式下可涂抹机械手区域"],
  "upload": ["Upload image", "上传图片"],
  "camera": ["Use camera", "相机采集"],
  "capture": ["Capture frame", "拍摄这一帧"],
  "closeCamera": ["Close camera", "关闭相机"],
  "instruction": ["Task instruction", "任务指令"],
  "instructionPlaceholder": ["Example: Grasp the cup with the right hand and slowly lift it.", "例如：用右手抓住桌上的杯子，缓慢抬起。"],
  "hfov": ["Camera horizontal FOV (°)", "相机水平视场角（°）"],
  "pitch": ["Fixed camera pitch (°)", "固定相机俯角（°）"],
  "cameraPlaceholder": ["Enter camera FOV", "填写相机参数"],
  "pitchPlaceholder": ["Enter measured pitch", "填写测量值"],
  "cameraHint": ["Fixed-camera mode assumes a stationary camera during capture and generation. Settings are saved in this browser.", "固定相机模式：拍摄与生成期间相机保持不动。参数会保存在本浏览器。"],
  "advanced": ["Edit region & calibration", "编辑区域与标定选项"],
  "manualMask": ["Specify the edit region manually (automatic by default)", "手动指定换手区域（默认自动识别）"],
  "maskHint": ["Paint the robot hand, wrist hardware, and visible arm plus the intended human outline, avoiding objects. White areas allow edits.", "涂抹机械手、腕部连接件和可见机械臂及预期人手臂轮廓，避开物体。白色区域允许编辑。"],
  "brush": ["Brush", "笔刷"],
  "clear": ["Clear", "清除"],
  "uploadMask": ["Or upload a black-and-white mask", "或上传黑白 mask"],
  "editedFile": ["Existing human-hand edit (optional; skips image editing)", "已有的人手编辑图（可选，跳过图片编辑服务）"],
  "cameraTransform": ["Camera → demonstration transform (optional JSON override)", "相机 → 示范坐标系变换（可选 JSON 覆盖）"],
  "placement": ["Demonstration → robot-world calibration (optional 4×4 JSON)", "示范 → 机器人世界坐标标定（可选，4×4 JSON）"],
  "placementHint": ["Without robot calibration, the trajectory aligns to the repository's initial hand pose for an experimental replay.", "未提供机器人标定时，轨迹对齐到仓库的初始手位，属于姿态回放实验。"],
  "seed": ["Random seed", "随机种子"],
  "generate": ["Generate 4-second trajectory", "生成 4 秒动作轨迹"],
  "automaticHint": ["Headless Codex edits the hands; video/action generation and retargeting run automatically.", "后台通过无界面 Codex 换手，并完成视频、动作生成和 retarget。"],
  "results": ["Generated results", "生成结果"],
  "waiting": ["Waiting for input", "等待输入"],
  "maskStage": ["Locate hands", "定位机械手"],
  "editStage": ["Edit hands", "换成人手"],
  "upliftStage": ["Estimate pose", "提取初始姿态"],
  "generateStage": ["Generate demo", "生成示范"],
  "retargetStage": ["Retarget", "动作映射"],
  "emptyTitle": ["Every motion starts with one frame.", "每段动作，从这一帧开始。"],
  "emptyHint": ["The edited hands, generated video and robot joint trajectories will appear here.", "这里会显示编辑后的人手、生成视频和机器人关节轨迹。"],
  "importNotice": ["Saved regression result: hand editing and H3 video generation were not rerun.", "已保存结果的回归展示：没有重新执行换手或 H3 视频生成。"],
  "sourceAlt": ["Original robot view", "原始机器人视角"],
  "sourceCaption": ["Robot initial observation", "机器人初始观察"],
  "editedAlt": ["Edited human-hand view", "编辑后的人手视角"],
  "editPending": ["Waiting for hand edit", "等待人手编辑"],
  "editedCaption": ["Human-hand initial condition", "人手初始条件"],
  "viewMask": ["View actual edit region", "查看实际编辑区域"],
  "maskAlt": ["White indicates the allowed edit region", "白色为允许编辑区域"],
  "jointTrajectory": ["Joint trajectories", "关节角轨迹"],
  "jointGroup": ["Select a joint group", "选择关节组"],
  "armOption": ["UR5e · 6 joints", "UR5e · 6 个关节"],
  "handOption": ["DG5F · 20 joints", "DG5F · 20 个关节"],
  "chartAlt": ["Joint angle over time", "关节角随时间的曲线"],
  "chartHint": ["Horizontal: seconds · Vertical: radians", "横轴：秒 · 纵轴：弧度"],
  "executionNote": ["Exported trajectory; not executed on hardware. The repository's real_robot/replay_real.py reads the NPZ. JSON includes the existing controllers' arm paths and hand targets.", "轨迹已导出，尚未在硬件执行。NPZ 可由仓库的 real_robot/replay_real.py 读取；JSON 包含现有控制器的路径和手部目标消息。"],
  "recent": ["Recent jobs", "最近任务"],
  "refresh": ["Refresh", "刷新"],
  "noJobs": ["No jobs yet.", "还没有任务。"],
  "checking": ["● Checking services", "● 正在检查服务"],
  "ready": ["● Services ready", "● 服务已就绪"],
  "notReady": ["● Services not ready", "● 服务尚未就绪"],
  "unreachable": ["● Backend unavailable", "● 无法连接后端"],
  "imageUnavailable": ["Headless image editing / masking unavailable: check Codex login and the operator prompt file.", "图片编辑／自动 mask 尚未配置 GEMINI_API_KEY"],
  "upliftUnavailable": ["WiLoR is not configured or its environment cannot run.", "WiLoR 环境不可执行或尚未配置"],
  "h3Unavailable": ["H3 is not ready: mirrored-12983 and the custom first-frame endpoint are required.", "H3 未就绪：需要 mirrored-12983 及自定义首帧接口"],
  "queued": ["Queued", "排队中"],
  "running": ["Running", "运行中"],
  "completed": ["Completed", "已完成"],
  "failed": ["Failed", "失败"],
  "interrupted": ["Interrupted", "已中断"],
  "durationMetric": ["Duration", "轨迹时长"],
  "samplesMetric": ["Joint samples", "关节采样"],
  "errorMetric": ["Solver fingertip error", "求解器指尖误差"],
  "alignmentMetric": ["Spatial alignment", "空间对齐"],
  "calibrated": ["Calibration supplied", "已提供标定"],
  "experimental": ["Initial pose · experimental", "初始手位 · 实验"],
  "downloadAll": ["Download all results ↓", "下载全部结果 ↓"],
  "downloadNPZ": ["Robot NPZ", "机器人 NPZ"],
  "downloadCSV": ["CSV", "CSV"],
  "downloadTrajectory": ["Trajectory JSON", "轨迹 JSON"],
  "downloadCommands": ["Controller messages", "控制器消息"],
  "downloadManifest": ["Run manifest", "运行记录"],
  "chooseFile": ["Choose file", "选择文件"],
  "noFile": ["No file selected", "未选择文件"],
  "readImageError": ["Could not read the image.", "无法读取图片"],
  "imageSizeError": ["Choose an image smaller than 16 MiB.", "请选择小于 16 MiB 的图片"],
  "megapixelError": ["Images must not exceed 16 MP.", "图片不能超过 16 MP"],
  "uploadFirst": ["Upload a robot image first.", "请先上传机器人图片"],
  "maskSizeError": ["Mask dimensions must match the original image.", "mask 尺寸必须与原图相同"],
  "editSizeError": ["Edited image dimensions must match the original image.", "编辑图尺寸必须与原图相同"],
  "cameraError": ["Camera access requires localhost or HTTPS. Open this page through the SSH tunnel.", "相机需要 localhost 或 HTTPS；请通过 SSH 隧道访问本页"],
  "paintError": ["Paint the robot-hand region and leave some area protected.", "请涂抹机械手区域，并保留未编辑区域"],
  "inputError": ["Upload or capture a robot image first.", "请先上传或拍摄机器人图片"],
  "editConfigError": ["Image editing is not configured: set GEMINI_API_KEY on the server", "图片编辑尚未配置：请在服务器设置 GEMINI_API_KEY"],
  "maskConfigError": ["Automatic masking is not configured; upload a mask", "自动 mask 尚未配置，请上传 mask"],
  "upliftConfigError": ["WiLoR interpreter and script must be configured on the server", "请在服务器配置 WiLoR 解释器和脚本"],
  "queueFull": ["Queue full; wait for an existing job", "队列已满，请等待已有任务完成"],
  "restartError": ["Server restarted; job was not resubmitted.", "服务器已重启，任务未自动重新提交。"]
};
let language = "en";
try { if (localStorage.getItem("retarget-language") === "zh") language = "zh"; } catch {}
function t(key) { return messages[key]?.[language === "zh" ? 1 : 0] ?? key; }
function localize(message) {
  const entry = Object.entries(messages).find(([, values]) => values.includes(message));
  return entry ? t(entry[0]) : message;
}
function renderFileLabels() {
  document.querySelectorAll("[data-file-name]").forEach(el => {
    const file = document.getElementById(el.dataset.fileName).files[0];
    el.textContent = file ? file.name : t("noFile");
  });
}
function setLanguage(value) {
  language = value === "zh" ? "zh" : "en";
  try { localStorage.setItem("retarget-language", language); } catch {}
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.title = t("pageTitle");
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  for (const attribute of ["placeholder", "alt", "aria-label"]) {
    document.querySelectorAll(`[data-i18n-${attribute}]`).forEach(el => {
      el.setAttribute(attribute, t(el.getAttribute(`data-i18n-${attribute}`)));
    });
  }
  const button = document.getElementById("language-toggle");
  button.textContent = language === "en" ? "中文" : "English";
  button.setAttribute("aria-label", language === "en" ? "Switch to Chinese" : "切换到英文");
  button.lang = language === "en" ? "zh-CN" : "en";
  renderFileLabels();
  window.dispatchEvent(new Event("languagechange"));
}
document.getElementById("language-toggle").onclick = () => setLanguage(language === "en" ? "zh" : "en");
document.querySelectorAll("[data-file-button]").forEach(button => {
  button.onclick = () => document.getElementById(button.dataset.fileButton).click();
});
document.querySelectorAll("input[type=file]").forEach(input => input.addEventListener("change", () => {
  setTimeout(renderFileLabels, 0);
}));
setLanguage(language);

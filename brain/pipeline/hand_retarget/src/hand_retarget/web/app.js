"use strict";
const $ = id => document.getElementById(id);
let inputImage = null, inputB64 = null, editedB64 = null, currentJob = null;
let futurePollTimer = null, futureJobId = null;
let pollTimer = null, stream = null, trajectory = null, drawing = false, previous = null;
const mask = document.createElement("canvas");
let lastJob = null, lastJobs = [], formError = "", jobError = "";
let robotCapture = null, cropSource = null, cropSourceImage = null, cropDirty = false;
let captureInitialization = Promise.resolve();
let diffusionState=null,diffusionPolling=false,diffusionInitialized=false;
let serverAction=null,lastServerStatusKey='',serverEvents=[];
const stages = ["mask", "edit", "uplift", "generate", "retarget"];

function error(message) { formError = message; $("form-error").textContent = localize(message); $("form-error").hidden = !message; }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}
function imageFromURL(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(t("readImageError")));
    image.src = url;
  });
}
async function fileImage(file) {
  if (!file || file.size > 16*1024*1024) throw new Error(t("imageSizeError"));
  const url = URL.createObjectURL(file);
  try { return await imageFromURL(url); } finally { URL.revokeObjectURL(url); }
}
function encodeImage(image) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  if (canvas.width * canvas.height > 16000000) throw new Error(t("megapixelError"));
  canvas.getContext("2d").drawImage(image, 0, 0);
  return canvas.toDataURL("image/png").split(",")[1];
}
async function setInput(image) {
  robotCapture = null; $("capture-result").hidden = true;
  inputB64 = encodeImage(image); inputImage = image;
  $("input-preview").width = mask.width = image.naturalWidth || image.width;
  $("input-preview").height = mask.height = image.naturalHeight || image.height;
  $("input-preview").hidden = false; $("upload-prompt").hidden = true;
  $("image-size").textContent = `${mask.width} × ${mask.height}`;
  $("mask-file").value = ""; $("edited-file").value = ""; editedB64 = null;
  renderFileLabels();
  drawInput(); error("");
}
function drawInput() {
  if (!inputImage) return;
  const canvas = $("input-preview"), ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(inputImage, 0, 0, canvas.width, canvas.height);
  if ($("manual-mask").checked) { ctx.globalAlpha = .55; ctx.drawImage(mask, 0, 0); ctx.globalAlpha = 1; }
}
function paint(event) {
  if (!drawing || !inputImage) return;
  const rect = $("input-preview").getBoundingClientRect();
  const p = [(event.clientX-rect.left)*mask.width/rect.width, (event.clientY-rect.top)*mask.height/rect.height];
  const ctx = mask.getContext("2d"); ctx.strokeStyle = "white"; ctx.fillStyle = "white";
  ctx.lineWidth = Number($("brush").value)*mask.width/rect.width; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(...(previous || p)); ctx.lineTo(...p); ctx.stroke();
  ctx.beginPath(); ctx.arc(...p, ctx.lineWidth/2, 0, Math.PI*2); ctx.fill();
  previous = p; drawInput();
}
$("input-preview").addEventListener("pointerdown", event => {
  if (!$("manual-mask").checked) return;
  drawing = true; previous = null; $("input-preview").setPointerCapture(event.pointerId); paint(event);
});
$("input-preview").addEventListener("pointermove", paint);
for (const name of ["pointerup", "pointercancel"]) $("input-preview").addEventListener(name, () => {drawing=false;previous=null;});
$("manual-mask").onchange = () => { $("mask-tools").hidden = !$("manual-mask").checked; drawInput(); };
$("clear-mask").onclick = () => {mask.getContext("2d").clearRect(0,0,mask.width,mask.height);drawInput();};
$("choose-image").onclick = () => $("image-file").click();
$("capture-robot").onclick = async () => {
  const button = $("capture-robot");
  button.disabled = true; $("generate").disabled = true; error("");
  try {
    await captureInitialization;
    stopCamera();
    const captured = await api("/api/captures", {method:"POST", headers:{"X-Retarget-Client":"1"}});
    cropSource = captured;
    cropSourceImage = await imageFromURL(captured.files["undistorted.png"]);
    await showRobotCapture(captured);
    if(!await restoreSavedCropDefaults())resetCropControls();
    await applyRobotCrop();
  } catch(e) {error(e.message);} finally {button.disabled = false; $("generate").disabled = false;}
};
async function showRobotCapture(captured) {
  await setInput(await imageFromURL(captured.files["undistorted.png"]));
  robotCapture = captured;
  refreshCalibrationStatus();
  $("capture-result").hidden = false;
  $("capture-robot-overlay").src = captured.files["overlay.png"];
  $("capture-mano-overlay").src = captured.files["mano_overlay.png"];
  $("capture-json").href = captured.files["hand_joints.json"];
  $("capture-image").href = captured.files["undistorted.png"];
  const camera = captured.camera, k = camera.undistorted_intrinsics.matrix;
  $("capture-summary").textContent = `${camera.width} × ${camera.height} · 21 MANO points · K: fx=${k[0][0].toFixed(2)}, fy=${k[1][1].toFixed(2)}, cx=${k[0][2].toFixed(2)}, cy=${k[1][2].toFixed(2)}`;
}
function cropParameters() {
  if (!cropSource) throw new Error("Capture a robot image first.");
  const ow=Number($("crop-width").value), oh=Number($("crop-height").value);
  if(!Number.isInteger(ow)||!Number.isInteger(oh)||ow<32||oh<32||ow>4096||oh>4096)throw new Error("Output dimensions must be integers between 32 and 4096.");
  const sw=cropSource.camera.width, sh=cropSource.camera.height;
  const ratio=ow/oh, zoom=Number($("crop-zoom").value);
  const height=Math.min(sh,sw/ratio)/zoom, width=height*ratio;
  return {x:(sw-width)*Number($("crop-x").value)/100,
    y:(sh-height)*Number($("crop-y").value)/100,width,height,output_width:ow,output_height:oh};
}
function renderCropPreview() {
  if(!cropSourceImage)return;
  try {
    const p=cropParameters(), canvas=$("crop-preview");
    canvas.width=cropSource.camera.width;canvas.height=cropSource.camera.height;
    const ctx=canvas.getContext("2d");ctx.drawImage(cropSourceImage,0,0);
    ctx.fillStyle="rgba(0,0,0,.5)";ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(cropSourceImage,p.x,p.y,p.width,p.height,p.x,p.y,p.width,p.height);
    ctx.strokeStyle="#ffd43b";ctx.lineWidth=4;ctx.strokeRect(p.x,p.y,p.width,p.height);
    $("crop-description").textContent=`${p.output_width} × ${p.output_height} · source crop ${p.width.toFixed(1)} × ${p.height.toFixed(1)} at (${p.x.toFixed(1)}, ${p.y.toFixed(1)})`;
  } catch(e) {error(e.message);}
}
function markCropDirty() {
  cropDirty=true;$("convert-hand").disabled=true;renderCropPreview();
}
function resetCropControls() {
  $("crop-zoom").value=1;$("crop-y").value=50;
  const points=cropSource.mano.joints.filter(j=>j.pixel_undistorted&&j.in_image);
  const handX=points.length?points.reduce((v,j)=>v+j.pixel_undistorted[0],0)/points.length:cropSource.camera.width/2;
  const handY=points.length?points.reduce((v,j)=>v+j.pixel_undistorted[1],0)/points.length:cropSource.camera.height/2;
  const p=cropParameters(), sw=cropSource.camera.width, sh=cropSource.camera.height;
  // Bias toward the hand and the table on its left, trimming the wall on the right.
  const centerX=.5*handX+.5*sw*.4, centerY=.5*handY+.5*sh*.5;
  $("crop-x").value=sw>p.width?Math.max(0,Math.min(100,(centerX-p.width/2)/(sw-p.width)*100)):50;
  $("crop-y").value=sh>p.height?Math.max(0,Math.min(100,(centerY-p.height/2)/(sh-p.height)*100)):50;
  markCropDirty();
}
async function applyRobotCrop() {
  const button=$("apply-crop");button.disabled=true;$("convert-hand").disabled=true;
  try {
    const parameters=cropParameters();
    const defaults={version:1,width:parameters.output_width,height:parameters.output_height,
      zoom:Number($("crop-zoom").value),x:Number($("crop-x").value),y:Number($("crop-y").value)};
    const captured=await api(`/api/captures/${cropSource.id}/crop`,{method:"POST",headers:{"Content-Type":"application/json","X-Retarget-Client":"1"},body:JSON.stringify({...parameters,defaults})});
    await showRobotCapture(captured);
    cropDirty=JSON.stringify(parameters)!==JSON.stringify(cropParameters());
    $("convert-hand").disabled=cropDirty;error("");
    try {localStorage.setItem("retarget-crop-defaults",JSON.stringify(defaults));} catch {}
  } finally {button.disabled=false;}
}
for(const id of ["crop-width","crop-height","crop-zoom","crop-x","crop-y"])$(id).oninput=markCropDirty;
$("reset-crop").onclick=()=>{if(cropSource)resetCropControls();};
$("apply-crop").onclick=async()=>{try{await applyRobotCrop();}catch(e){error(e.message);}};
function restoreCropDefaults() {
  try {
    const saved=JSON.parse(localStorage.getItem("retarget-crop-defaults")||"null");
    if(!saved||saved.version!==1||!Number.isInteger(saved.width)||!Number.isInteger(saved.height)||
      saved.width<32||saved.width>4096||saved.height<32||saved.height>4096||
      !Number.isFinite(saved.zoom)||saved.zoom<1||saved.zoom>4||
      !Number.isFinite(saved.x)||saved.x<0||saved.x>100||!Number.isFinite(saved.y)||saved.y<0||saved.y>100)return false;
    for(const [id,key]of [["crop-width","width"],["crop-height","height"],["crop-zoom","zoom"],["crop-x","x"],["crop-y","y"]])$(id).value=saved[key];
    if(cropSource)markCropDirty();
    return true;
  }catch{return false;}
}
async function restoreSavedCropDefaults() {
  try {
    const response=await api('/api/crop-defaults');
    if(response.defaults) {
      // Keep a browser fallback while the server remains the persistent default.
      try {localStorage.setItem('retarget-crop-defaults',JSON.stringify(response.defaults));}catch{}
      const saved=response.defaults;
      for(const [id,key]of [["crop-width","width"],["crop-height","height"],["crop-zoom","zoom"],["crop-x","x"],["crop-y","y"]])$(id).value=saved[key];
      if(cropSource)markCropDirty();return true;
    }
  }catch{}
  return restoreCropDefaults();
}
async function initializeSavedCapture() {
  await restoreSavedCropDefaults();
  try {
    const captured=await api('/api/captures/latest');
    let source=captured;
    while(source.source_capture_id)source=await api(`/api/captures/${source.source_capture_id}`);
    cropSource=source;
    cropSourceImage=await imageFromURL(source.files['undistorted.png']);
    await showRobotCapture(captured);
    const hasDefaults=await restoreSavedCropDefaults();
    if(!hasDefaults)resetCropControls();
    renderCropPreview();
    const p=cropParameters(), rectangle=captured.crop?.rectangle_source_px;
    const matches=rectangle&&[p.x,p.y,p.width,p.height].every((value,i)=>Math.abs(value-rectangle[i])<1e-6)
      &&captured.camera.width===p.output_width&&captured.camera.height===p.output_height;
    if(matches){cropDirty=false;$("convert-hand").disabled=false;}
    else await applyRobotCrop();
  } catch(e) {
    if(e.message!=="No saved capture yet.")error(e.message);
  }
}
if(!restoreCropDefaults()) {
  try {const saved=JSON.parse(localStorage.getItem("retarget-crop-output")||"null");if(saved){$("crop-width").value=saved.width;$("crop-height").value=saved.height;}}catch{}
}
$("convert-hand").onclick = async () => {
  if (!robotCapture || cropDirty) return;
  const button = $("convert-hand"); button.disabled = true; error("");
  try {
    const request = {capture_id:robotCapture.id, stop_after:"edit",
      instruction:$("instruction").value.trim() || "Replace the robot hand and visible arm with a human right hand and arm aligned to the captured MANO skeleton.",
      mask_b64:$("manual-mask").checked?encodeMask():null};
    const job = await api("/api/jobs",{method:"POST",headers:{"Content-Type":"application/json","X-Retarget-Client":"1"},body:JSON.stringify(request)});
    await selectJob(job.id); await history();
  } catch(e) {error(e.message);} finally {button.disabled = false;}
};
$("dropzone").onclick = () => { if (!inputImage) $("image-file").click(); };
$("image-file").onchange = async event => {try {await setInput(await fileImage(event.target.files[0]));} catch(e){error(e.message);}};
$("dropzone").ondragover = event => {event.preventDefault();$("dropzone").classList.add("drag");};
$("dropzone").ondragleave = () => $("dropzone").classList.remove("drag");
$("dropzone").ondrop = async event => {
  event.preventDefault();$("dropzone").classList.remove("drag");
  try {await setInput(await fileImage(event.dataTransfer.files[0]));} catch(e){error(e.message);}
};
$("mask-file").onchange = async event => {
  try {
    if (!inputImage) throw new Error(t("uploadFirst"));
    const image = await fileImage(event.target.files[0]);
    if (image.naturalWidth !== mask.width || image.naturalHeight !== mask.height) throw new Error(t("maskSizeError"));
    const ctx = mask.getContext("2d");ctx.clearRect(0,0,mask.width,mask.height);ctx.drawImage(image,0,0);
    const data = ctx.getImageData(0,0,mask.width,mask.height);
    for(let i=0;i<data.data.length;i+=4){const v=data.data[i]>=128;data.data[i]=data.data[i+1]=data.data[i+2]=255;data.data[i+3]=v?255:0;}
    ctx.putImageData(data,0,0);drawInput();error("");
  } catch(e){error(e.message);}
};
$("edited-file").onchange = async event => {
  editedB64 = null;
  if (!event.target.files.length) return;
  try {
    if (!inputImage) throw new Error(t("uploadFirst"));
    const image = await fileImage(event.target.files[0]);
    if(image.naturalWidth!==mask.width || image.naturalHeight!==mask.height) throw new Error(t("editSizeError"));
    editedB64=encodeImage(image);error("");
  } catch(e){event.target.value="";renderFileLabels();error(e.message);}
};
function stopCamera() {if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;$("camera-area").hidden=true;}
$("open-camera").onclick = async () => {
  try {
    stopCamera();
    if (!navigator.mediaDevices) throw new Error(t("cameraError"));
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});
    $("camera").srcObject=stream;$("camera-area").hidden=false;
  } catch(e){error(e.message);}
};
$("close-camera").onclick=stopCamera;
$("capture").onclick=async()=>{
  const video=$("camera");
  if(!video.videoWidth)return;
  const canvas=document.createElement("canvas");canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  canvas.getContext("2d").drawImage(video,0,0);
  try{await setInput(await imageFromURL(canvas.toDataURL("image/png")));stopCamera();}catch(e){error(e.message);}
};
window.addEventListener("pagehide",stopCamera);
async function jsonFile(id){const file=$(id).files[0];return file?JSON.parse(await file.text()):null;}
function encodeMask(){
  const canvas=document.createElement("canvas");canvas.width=mask.width;canvas.height=mask.height;
  const ctx=canvas.getContext("2d");ctx.fillStyle="black";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(mask,0,0);
  const data=ctx.getImageData(0,0,canvas.width,canvas.height);let edited=0;
  for(let i=0;i<data.data.length;i+=4){const v=data.data[i]>=128?255:0;edited+=v>0;data.data[i]=data.data[i+1]=data.data[i+2]=v;data.data[i+3]=255;}
  if(!edited || edited===canvas.width*canvas.height)throw new Error(t("paintError"));
  ctx.putImageData(data,0,0);return canvas.toDataURL("image/png").split(",")[1];
}
$("generate-form").onsubmit=async event=>{
  event.preventDefault();error("");$("generate").disabled=true;
  try {
    if(!inputB64)throw new Error(t("inputError"));
    if(robotCapture&&cropDirty)throw new Error("Apply the updated crop before continuing.");
    const camera=await jsonFile("camera-transform");
    if(!robotCapture)throw new Error("Use Capture robot camera to supply calibrated intrinsics for this workflow.");
    const request={image_b64:inputB64,instruction:$("instruction").value,
      seed:Number($("seed").value),
      camera_to_demo:camera,robot_from_demo:await jsonFile("placement"),
      mask_b64:$("manual-mask").checked?encodeMask():null,edited_image_b64:editedB64};
    if (robotCapture) {delete request.image_b64; request.capture_id = robotCapture.id;}
    const job=await api("/api/jobs",{method:"POST",headers:{"Content-Type":"application/json","X-Retarget-Client":"1"},body:JSON.stringify(request)});
    await selectJob(job.id);await history();
  }catch(e){error(e.message);}finally{$("generate").disabled=false;}
};
function renderHealth(){
  const s=diffusionState,button=$("health");
  const labels={down:'GPU stopped',stopped:'GPU stopped',queued:'GPU queued',booting:'GPU loading',ready:'GPU ready',failed:'GPU startup failed',cancelled:'GPU cancelled',completed:'GPU job ended'};
  button.textContent='● '+(serverAction?(serverAction==='start'?'Submitting GPU job…':'Connecting to CSCS…'):!s?'Checking CSCS…':!s.connected?'CSCS disconnected':s.slurm_state==='TIMEOUT'?'GPU allocation expired':labels[s.state]||`GPU ${s.state}`);
  button.dataset.state=serverAction?'busy':s?.connected?s.state:'down';
  button.className=`pill ${s?.state==='ready'&&s.connected?'ready':''}`;
  button.title=`Server startup, checkpoint selection and boot log${s?.jobid?' · job '+s.jobid:''}`;
  $("health-detail").hidden=true;
}
$("health").onclick=()=>{$("server-dialog").showModal();refreshDiffusion();};
$("close-server-dialog").onclick=()=>$("server-dialog").close();
$("server-dialog").addEventListener('click',event=>{if(event.target===$("server-dialog")){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
function resetResult(){
  trajectory=null;lastJob=null;jobError="";$("result").hidden=false;$("empty-state").hidden=true;
  for(const id of ["edited-image","output-video","trajectory","execution-note","mask-result","converted-pose","job-review","predicted-hand-result"])$(id).hidden=true;
  $("output-video").pause();$("output-video").removeAttribute("src");$("output-video").load();
  $("edit-pending").hidden=false;$("metrics").replaceChildren();$("downloads").replaceChildren();$("job-error").hidden=true;
  $("import-notice").hidden=true;$("steps").hidden=false;
}
async function selectJob(id){
  if(lastJobs.find(job=>job.id===id)?.mode==="robot_rollout"){window.selectRolloutJob?.(id);return;}
  if(lastJobs.find(job=>job.id===id)?.mode==="mujoco_replay"){window.selectSimulationJob?.(id);return;}
  if(lastJobs.find(job=>job.id===id)?.mode==="generated_hand_retarget"){window.selectRetargetJob?.(id);return;}
  clearTimeout(pollTimer);currentJob=id;localStorage.setItem("retarget-last-job",id);resetResult();
  await poll(id);
}
async function poll(id){
  try{
    const job=await api(`/api/jobs/${id}`);if(currentJob!==id)return;
    renderJob(job);
    if(["queued","running"].includes(job.status))pollTimer=setTimeout(()=>poll(id),1800);
    else await history();
  }catch(e){if(currentJob===id){jobError=e.message;$("job-error").textContent=localize(jobError);$("job-error").hidden=false;}}
}
function metric(label,value){const div=document.createElement("div");div.className="metric";const title=document.createElement("small");title.textContent=label;div.append(title,document.createTextNode(value));$("metrics").append(div);}
function renderJob(job){
  if(job.mode==="diffusion_video")renderFutureResult(job);
  lastJob=job;jobError=job.error||"";renderJobText(job);
  $("job-review").textContent=job.review_note||"";$("job-review").hidden=!job.review_note;
  $("source-image").closest(".previews").hidden=!job.input_url;
  if(job.input_url)$("source-image").src=job.input_url;
  $("import-notice").hidden=job.mode!=="imported_result";
  $("steps").hidden=job.mode==="imported_result";
  const active=job.mode==="diffusion_video"?3:stages.indexOf(job.progress?.stage);
  document.querySelectorAll("#steps li").forEach((el,i)=>{el.hidden=job.mode==="diffusion_video"?i!==3:job.mode==="capture_to_human"&&i>1;el.className=job.status==="completed"||i<active?"done":i===active?"active":"";});
  const files=job.artifacts||{};
  if(files["edited.png"]){$("edited-image").src=files["edited.png"];$("edited-image").hidden=false;$("edit-pending").hidden=true;}
  if(files["edit_mask.png"]){$("mask-image").src=files["edit_mask.png"];$("mask-result").hidden=false;}
  if(files["edited_mano_overlay.png"]){$("converted-mano-overlay").src=files["edited_mano_overlay.png"];$("converted-pose").hidden=false;}
  if(job.status!=="completed")return;
  if(files["video.mp4"]){$("output-video").src=files["video.mp4"];$("output-video").hidden=false;}
  if(files["predicted_hand_overlay.mp4"]){$("predicted-hand-video").src=files["predicted_hand_overlay.mp4"];$("predicted-hand-result").hidden=false;}
  $("execution-note").hidden=job.mode!=="image_to_robot";
  if(files["trajectory.json"]){const id=job.id;api(files["trajectory.json"]).then(data=>{if(currentJob===id){trajectory=data;$("trajectory").hidden=false;drawChart();}}).catch(e=>error(e.message));}
}
function renderJobText(job){
  $("job-status").textContent=t(job.status);$("job-instruction").textContent=job.instruction;
  $("job-error").textContent=localize(jobError);$("job-error").hidden=!jobError;
  if(job.status!=="completed")return;
  const files=job.artifacts||{};
  $("metrics").replaceChildren();
  if(job.mode==="capture_to_human") {metric(t("manoOverlay"),"21");}
  else if(job.mode==="diffusion_video"){metric(t("durationMetric"),`${((job.metrics?.frames||0)/(job.metrics?.fps||24)).toFixed(2)} s`);metric(t("predictedHandVideo"),`${job.metrics?.frames||0} frames · 21 × 3`);}
  else {metric(t("durationMetric"),`${(job.metrics?.duration_seconds??4).toFixed(2)} s`);metric(t("samplesMetric"),`${job.metrics?.exports?.samples??"—"} × 26`);}
  if(job.metrics?.mean_fingertip_error_m!=null)metric(t("errorMetric"),`${(job.metrics.mean_fingertip_error_m*1000).toFixed(2)} mm`);
  if(job.mode==="image_to_robot")metric(t("alignmentMetric"),t(job.metrics?.placement_mode==="calibrated"?"calibrated":"experimental"));
  $("downloads").replaceChildren();
  const labels={"bundle.zip":"downloadAll","robot_actions.npz":"downloadNPZ","robot_actions.csv":"downloadCSV","trajectory.json":"downloadTrajectory","controller_commands.json":"downloadCommands","manifest.json":"downloadManifest","capture.json":"captureJSON","capture_guidance.json":"captureJSON","edited_mano_overlay.png":"convertedPose","first_frame_gt.npz":"downloadConditioning","first_frame_gt.json":"downloadConditioningInfo","future_hands.npz":"downloadFutureHands","predicted_hand_overlay.mp4":"predictedHandVideo"};
  for(const [name,label]of Object.entries(labels)){if(files[name]){const a=document.createElement("a");a.href=files[name];a.download=name;a.textContent=t(label);$("downloads").append(a);}}
}
function drawChart(){
  if(!trajectory)return;
  const canvas=$("chart"),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height,pad=38;
  ctx.clearRect(0,0,w,h);const arm=$("joint-group").value==="arm",start=arm?0:6,end=arm?6:26;
  let lo=Infinity,hi=-Infinity;for(const p of trajectory.points)for(let j=start;j<end;j++){lo=Math.min(lo,p.positions[j]);hi=Math.max(hi,p.positions[j]);}
  if(hi-lo<.1){lo-=.05;hi+=.05;}
  ctx.font="12px system-ui";ctx.fillStyle="#807b72";ctx.strokeStyle="#eeeae3";
  for(let i=0;i<5;i++){const y=pad+(h-2*pad)*i/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke();ctx.fillText((hi-(hi-lo)*i/4).toFixed(1),2,y+4);ctx.fillText(`${i}s`,pad+(w-2*pad)*i/4-7,h-8);}
  for(let j=start;j<end;j++){ctx.strokeStyle=`hsl(${(j-start)*137.5%360},45%,45%)`;ctx.lineWidth=1.5;ctx.beginPath();
    trajectory.points.forEach((p,i)=>{const x=pad+p.time_from_start/trajectory.duration_seconds*(w-2*pad),y=pad+(hi-p.positions[j])/(hi-lo)*(h-2*pad);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();}
}
$("joint-group").onchange=drawChart;
async function history(){
  try{
    lastJobs=await api("/api/jobs");renderHistory();
  }catch(e){error(e.message);}
}
function renderHistory(){
    window.refreshRetargetJobs?.(lastJobs);
    window.refreshSimulationJobs?.(lastJobs);
    window.refreshRolloutJobs?.(lastJobs);
    renderDiffusionSources();
    renderFutureResults();
    $("history-list").replaceChildren();
    if(!lastJobs.length){const p=document.createElement("p");p.className="muted";p.textContent=t("noJobs");$("history-list").append(p);}
    for(const job of lastJobs){const b=document.createElement("button");b.className="history-item";b.type="button";
      const prompt=document.createElement("span");prompt.className="prompt";prompt.textContent=job.instruction+(job.calibration_status ? ' · Calibration '+job.calibration_status.capture_calibration+(job.calibration_status.current ? '' : ' (historical)') : '');
      const date=document.createElement("small");date.textContent=new Date(job.created_at*1000).toLocaleString(language==="zh"?"zh-CN":"en-GB");
      const status=document.createElement("span");status.textContent=t(job.status);b.append(prompt,date,status);b.onclick=()=>selectJob(job.id);$("history-list").append(b);}
}
window.addEventListener("languagechange",()=>{
  error(formError);renderHealth();renderHistory();
  if(lastJob)renderJobText(lastJob);
  else if(jobError)$("job-error").textContent=localize(jobError);
});
$("refresh-history").onclick=history;
function serverFeedback(message,state='busy'){
  const box=$("server-status-banner");if(box){box.textContent=message;box.dataset.state=state;}
}
function serverEvent(message){
  serverEvents.push(`${new Date().toLocaleTimeString()} — ${message}`);
  serverEvents=serverEvents.slice(-15);
  if($("server-activity"))$("server-activity").textContent=serverEvents.join('\n');
}
function diffusionError(message){for(const id of ["diffusion-error","server-error"]){if($(id)){$(id).textContent=message||"";$(id).hidden=!message;}}if(message){serverFeedback(message,'error');serverEvent(message);}}
function renderFutureResults(){
  const select=$("future-result-job"),previous=futureJobId||select.value;
  const jobs=lastJobs.filter(job=>job.mode==="diffusion_video");
  $("future-results").hidden=!jobs.length;select.replaceChildren();
  for(const job of jobs){const option=document.createElement("option");option.value=job.id;option.textContent=`${new Date(job.created_at*1000).toLocaleString()} · ${job.id.slice(0,8)} · ${t(job.status)}`;select.append(option);}
  if(jobs.some(job=>job.id===previous))select.value=previous;
  const job=jobs.find(job=>job.id===select.value);if(job)renderFutureResult(job);
}
function durationText(seconds){
  const value=Math.max(0,Math.round(seconds));return value<60?`${value}s`:`${Math.floor(value/60)}m ${value%60}s`;
}
async function pollFutureResult(id){
  try{
    const job=await api(`/api/jobs/${id}`,{signal:AbortSignal.timeout(15000),cache:'no-store'});
    if(futureJobId!==id)return;
    renderFutureResult(job);
    if(!['queued','running'].includes(job.status)){
      lastJobs=lastJobs.map(previous=>previous.id===id?job:previous);renderHistory();
    }
  }catch(e){
    if(futureJobId!==id)return;
    $("future-result-status").textContent=`${id.slice(0,8)} · Status update unavailable: ${e.message}. Retrying…`;
    futurePollTimer=setTimeout(()=>pollFutureResult(id),2500);
  }
}
function renderFutureProgress(job){
  const active=['queued','running'].includes(job.status),progress=job.progress||{},remote=progress.remote||{};
  const phase=progress.stage==='generate'?(remote.phase||remote.status||'generate'):(progress.stage||job.status);
  const labels={prepare:'Preparing captured GT input',preflight:'Checking inference service',register:'Uploading first frame and hand GT',
    generate:'Waiting for generation progress',queued:'Waiting in queue',embedding:'Encoding first frame and prompt',
    sampling:'Generating video and hand 3D',decoding:'Decoding video',robot:'Rendering robot preview on CSCS',
    download:'Downloading video and hand 3D',validate:'Validating generated video and GT alignment',completed:'Generation complete',failed:'Generation failed'};
  $("future-progress").hidden=!active;
  const bar=$("future-progress-bar");
  const measured=phase==='sampling'&&Number.isFinite(remote.step)&&Number.isFinite(remote.total)&&remote.total>0;
  $("future-progress-stage").textContent=(labels[phase]||phase)+(measured?` · Step ${remote.step}/${remote.total}`:'');
  if(measured){bar.max=remote.total;bar.value=Math.max(0,Math.min(remote.step,remote.total));}
  else bar.removeAttribute('value');
  const timing=[];
  if(Number.isFinite(job.created_at))timing.push(`Elapsed ${durationText(Date.now()/1000-job.created_at)}`);
  if(Number.isFinite(remote.eta_s))timing.push(`Stage ETA ${durationText(remote.eta_s)}`);
  if(Number.isFinite(progress.updated_at)){
    const age=Math.max(0,Date.now()/1000-progress.updated_at);
    timing.push(`Progress updated ${durationText(age)} ago`);
    if(age>15)timing.push('Waiting for a new progress update');
  }
  timing.push('Checks automatically every 2 seconds');
  $("future-progress-timing").textContent=timing.join(' · ');
}
function renderFutureResult(job){
  clearTimeout(futurePollTimer);futureJobId=job.id;
  renderFutureProgress(job);
  if(['queued','running'].includes(job.status))futurePollTimer=setTimeout(()=>pollFutureResult(job.id),1800);
  $("future-results").hidden=false;
  if([...$("future-result-job").options].some(option=>option.value===job.id))$("future-result-job").value=job.id;
  $("future-result-status").textContent=`${job.id.slice(0,8)} · ${t(job.status)}${job.error?" · "+job.error:""}`;
  const files=job.status==="completed"?(job.artifacts||{}):{};
  for(const [id,figure,name] of [["future-rgb-video","future-rgb-figure","video.mp4"],["future-overlay-video","future-overlay-figure","predicted_hand_overlay.mp4"]]){
    const video=$(id),url=files[name]||"";$(figure).hidden=!url;
    if((video.getAttribute("src")||"")!==url){video.pause();if(url)video.src=url;else video.removeAttribute("src");video.load();}
  }
  const download=$("future-hands-download");download.hidden=!files["future_hands.npz"];
  if(!download.hidden)download.href=files["future_hands.npz"];
}
$("future-result-job").onchange=()=>selectJob($("future-result-job").value);
function renderDiffusionSources(){
  const select=$("diffusion-source"), previous=select.value;
  const sources=lastJobs.filter(job=>job.status==="completed"&&job.mode==="capture_to_human");
  select.replaceChildren();
  for(const job of sources){const option=document.createElement("option");option.value=job.id;option.textContent=`${new Date(job.created_at*1000).toLocaleString()} · ${job.id.slice(0,8)}`;select.append(option);}
  if(sources.some(job=>job.id===previous))select.value=previous;
  renderDiffusionSource();
}
function renderDiffusionSource(){
  const job=lastJobs.find(item=>item.id===$("diffusion-source").value),files=job?.artifacts||{};
  $("diffusion-source-preview").hidden=!job;
  if(job){$("diffusion-input-image").src=files["edited.png"];$("diffusion-input-gt").src=files["edited_mano_overlay.png"]||files["mano_overlay.png"];}
  $("diffusion-source-note").textContent=job?.review_note||"";
  $("prepare-conditioning").disabled=!job;
  $("conditioning-download").hidden=true;
  updateDiffusionButtons();
}
function updateDiffusionButtons(){
  const state=diffusionState,capability=state?.capture_gt_bridge;
  $("load-checkpoint").disabled=!!serverAction||!state?.connected||!capability?.capture_gt_v1||(!state?.terminal&&!['down','stopped'].includes(state?.state));
  $("stop-checkpoint").disabled=!!serverAction||!state?.connected||!!state?.terminal||['down','stopped'].includes(state?.state);
  $("connect-diffusion").disabled=!!serverAction;
  $("generate-future").disabled=!state?.connected||state.state!=="ready"||
    state.selected_profile!==$("checkpoint-profile").value||!$("diffusion-source").value||!capability?.capture_gt_v1;
  const profile=state?.profiles?.find(item=>item.id===$("checkpoint-profile").value);
  $("checkpoint-recipe").textContent=profile?`${profile.inference_steps||50} steps · CFG ${profile.inference_cfg||5} · video/hand shift ${profile.inference_video_shift??'—'}/${profile.inference_hand_shift??'—'} · matched model base + hand codec + statistics`:'';
  renderHealth();
}
async function refreshDiffusion(){
  if(diffusionPolling)return diffusionPolling;
  diffusionPolling=(async()=>{
  try{
    const state=await api('/api/diffusion/status',{signal:AbortSignal.timeout(30000),cache:'no-store'});diffusionState=state;
    if(!diffusionInitialized){if(!$("aeon-url").value.trim())$("aeon-url").value=state.url||"http://127.0.0.1:8017";diffusionInitialized=true;}
    const select=$("checkpoint-profile"),previous=select.value;select.replaceChildren();
    for(const profile of state.profiles||[]){const option=document.createElement('option');option.value=profile.id;option.textContent=profile.label||profile.id;
      option.disabled=state.capture_gt_bridge?!state.capture_gt_bridge.profiles.includes(profile.id):profile.supports_gt_upload===false;select.append(option);}
    const options=[...select.options].filter(option=>!option.disabled).map(option=>option.value);
    select.value=options.includes(previous)?previous:options.includes(state.selected_profile)?state.selected_profile:options.includes('fast-align-v2-hf15-precision-29k')?'fast-align-v2-hf15-precision-29k':options[0]||'';
    const gpuStopped=['down','stopped'].includes(state.state);
    const expired=state.terminal&&state.slurm_state==='TIMEOUT';
    const expiryMessage='GPU allocation reached its time limit. The CSCS connection is working. Click Start server / load checkpoint to request a new allocation; Connect / refresh only checks status.';
    const gpuFailed=!expired&&['failed','cancelled','completed'].includes(state.state);
    $("diffusion-state").textContent=state.connected?`${t('serverConnected')} · ${expired?'GPU allocation expired':gpuStopped?t('gpuStopped'):state.state}`:'Not connected';
    $("server-active-profile").textContent=state.state==='ready'?(state.running_profile||state.selected_profile||'—'):'—';
    $("server-job").textContent=state.jobid||'—';$("server-node").textContent=state.node||'—';$("server-queue").textContent=state.selected_partition||state.partition||'—';
    $("diffusion-detail").textContent=expired?expiryMessage:state.load_error||state.error||(state.connected&&!state.capture_gt_bridge?'Install the capture-GT bridge in the connected aeon-webui checkout before loading/generating.':!state.connected?'Connect a running aeon-webui service or its local SSH tunnel. Checkpoints shown below are a catalog, not locally loaded weights.':gpuStopped?t('serverConnectedNext'):'');
    const statusMessage=!state.connected?`Connection failed: ${state.error||'Enter the aeon-webui URL and click Connect.'}`:
      !state.capture_gt_bridge?.capture_gt_v1?'Connected to CSCS, but the GT-input extension is unavailable.':
      expired?`Connected to CSCS ✓\nGPU allocation expired${state.jobid?' · job '+state.jobid:''}. ${expiryMessage}`:
      gpuStopped?'Connected to CSCS ✓\nGPU server is stopped. Select a checkpoint, then click Start server / load checkpoint.':
      gpuFailed?`GPU server ${state.state}${state.jobid?' · job '+state.jobid:''}\n${state.load_error||'Check the boot log below. Click Start server / load checkpoint to request a new allocation.'}`:
      state.state==='ready'?`GPU server ready ✓\n${state.running_profile||state.selected_profile||''}`:
      `Connected to CSCS ✓\n${state.state==='queued'?'Queued in Slurm — waiting for GPU allocation':state.state==='booting'?'Loading checkpoint on GPU server':state.state}${state.jobid?' · job '+state.jobid:''}${state.node?' · '+state.node:''}`;
    if(!serverAction)serverFeedback(statusMessage,!state.connected||!state.capture_gt_bridge?.capture_gt_v1||gpuFailed?'error':state.state==='ready'?'ready':gpuStopped||expired?'stopped':'busy');
    const bootLog=$("server-boot-log"),followLog=bootLog.scrollHeight-bootLog.scrollTop<=bootLog.clientHeight+20;
    bootLog.textContent=state.log_tail||(state.state==='queued'?'Waiting for Slurm allocation. Boot output appears when the GPU node starts.':state.state==='booting'?'GPU node allocated. Waiting for the first boot-log lines…':state.state==='ready'?'Model loaded. The GPU server reports ready.':'No boot log available for this server.');
    if(followLog)bootLog.scrollTop=bootLog.scrollHeight;
    const runtime=[];if(state.time_left)runtime.push(`${state.time_left} walltime left`);if(state.connected&&!gpuStopped)runtime.push(state.sol_enabled?'NVIDIA Sol enabled':'Dense inference');
    if(state.refiner_loaded!==undefined)runtime.push(`Prompt planner: ${state.refiner_loaded?'ready':state.refiner_error?'failed':'loading'}`);if(state.refiner_error)runtime.push(state.refiner_error);
    $("server-runtime-detail").textContent=runtime.join(' · ');
    const step=!state.connected?0:state.state==='queued'?1:state.state==='booting'?2:state.state==='ready'?4:1;
    document.querySelectorAll('#server-boot-steps li').forEach((el,i)=>{el.className=i<step?'done':i===step&&!gpuFailed&&!expired?'active':'';});
    if(state.connected&&['queued','booting','ready','failed'].includes(state.state)){$("checkpoint-queue").value=state.selected_partition||state.partition||'normal';}
    if($("server-last-check"))$("server-last-check").textContent=`Last checked: ${new Date().toLocaleTimeString()} · refreshes every 5 seconds`;
    const statusKey=JSON.stringify([state.connected,state.state,state.jobid,state.node,state.error]);
    if(statusKey!==lastServerStatusKey){serverEvent(statusMessage.replaceAll('\n',' '));lastServerStatusKey=statusKey;}
    updateDiffusionButtons();
    return state;
  }catch(e){diffusionState=null;diffusionError(`Status check failed: ${e.message}`);updateDiffusionButtons();return null;}finally{diffusionPolling=false;}
  })();
  return diffusionPolling;
}
$("diffusion-source").onchange=renderDiffusionSource;
$("checkpoint-profile").onchange=updateDiffusionButtons;
$("connect-diffusion").onclick=async()=>{
  if(serverAction)return;
  const button=$("connect-diffusion"),label=button.textContent;
  serverAction='connect';updateDiffusionButtons();button.textContent="Connecting…";
  diffusionError("");serverFeedback('Connecting to CSCS through '+$("aeon-url").value.trim()+'…');serverEvent('Connect clicked.');
  let failure=null;
  try{if(diffusionPolling)await diffusionPolling;await api('/api/diffusion/config',{method:'POST',headers:{'Content-Type':'application/json','X-Retarget-Client':'1'},body:JSON.stringify({url:$("aeon-url").value.trim()})});}
  catch(e){failure=e;}
  finally{serverAction=null;button.textContent=label;await refreshDiffusion();if(failure)diffusionError(failure.message);}
};
$("load-checkpoint").onclick=async()=>{
  if(serverAction)return;
  serverAction='start';updateDiffusionButtons();const button=$("load-checkpoint"),label=button.textContent;button.textContent='Submitting GPU job…';
  diffusionError("");serverFeedback('Submitting GPU allocation to CSCS…\nWaiting for Slurm to return a job ID.');serverEvent('Start clicked: '+$("checkpoint-profile").value);
  let failure=null;
  try{const result=await api('/api/diffusion/start',{method:'POST',headers:{'Content-Type':'application/json','X-Retarget-Client':'1'},body:JSON.stringify({profile:$("checkpoint-profile").value,partition:$("checkpoint-queue").value,sol_enabled:$("checkpoint-sol").checked})});serverEvent(`Start request accepted: ${result.submitted||result.state||'submitted'}${result.jobid?' · job '+result.jobid:''}`);}
  catch(e){failure=e;}
  finally{serverAction=null;button.textContent=label;await refreshDiffusion();if(failure)diffusionError(failure.message);}
};
$("stop-checkpoint").onclick=async()=>{diffusionError("");try{await api('/api/diffusion/stop',{method:'POST',headers:{'X-Retarget-Client':'1'}});await refreshDiffusion();}catch(e){diffusionError(e.message);}};
$("prepare-conditioning").onclick=async()=>{
  diffusionError("");$("prepare-conditioning").disabled=true;
  try{const result=await api(`/api/jobs/${$("diffusion-source").value}/conditioning`,{method:'POST',headers:{'X-Retarget-Client':'1'}});$("conditioning-download").href=result.bundle;$("conditioning-download").hidden=false;}
  catch(e){diffusionError(e.message);}finally{$("prepare-conditioning").disabled=false;}
};
$("generate-future").onclick=async()=>{
  diffusionError("");$("generate-future").disabled=true;
  try{const instruction=$("future-instruction").value.trim();if(!instruction)throw new Error('Describe the future action first.');
    const job=await api('/api/diffusion/generate',{method:'POST',headers:{'Content-Type':'application/json','X-Retarget-Client':'1'},body:JSON.stringify({source_job_id:$("diffusion-source").value,instruction,profile:$("checkpoint-profile").value,seed:Number($("seed").value),assume_zero_contact:$("zero-contact").checked})});await selectJob(job.id);await history();}
  catch(e){diffusionError(e.message);}finally{updateDiffusionButtons();}
};
history();refreshDiffusion();
setInterval(refreshDiffusion,5000);
captureInitialization=initializeSavedCapture();
const last=localStorage.getItem("retarget-last-job");if(last)selectJob(last);
if(new URLSearchParams(location.search).has('server'))$("server-dialog").showModal();

async function refreshCalibrationStatus() {
  const label = $("calibration-status");
  try {
    const state = await api('/api/calibration/status' + (robotCapture ? '?capture_id='+encodeURIComponent(robotCapture.id) : ''));
    label.textContent = !state.available ? 'Latest calibration unavailable: '+state.error+(robotCapture ? ' · Existing capture and downstream results retain their saved calibration.' : '')
      : 'Latest calibration: '+state.latest + (robotCapture
        ? (state.current ? ' · Current capture uses this calibration.' : ' · Displayed capture uses '+state.capture_calibration+'. All downstream stages use this capture’s calibration. New captures use the latest calibration.')
        : ' · New captures automatically use this calibration.');
    label.dataset.current = String(state.current === true);
  } catch(e) { label.textContent = 'Cannot check calibration: '+e.message; }
}
refreshCalibrationStatus();
setInterval(refreshCalibrationStatus, 5000);

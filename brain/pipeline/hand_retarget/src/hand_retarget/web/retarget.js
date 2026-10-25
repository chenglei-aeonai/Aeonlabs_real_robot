/* Offline calibrated DG5F trajectory preview. No hardware controls. */
(()=>{
  const el=id=>document.getElementById(id);
  let jobs=[],selected=null,pollTimer=null,data=null,loaded=null,frame=0,playing=false,lastTime=0;
  let yaw=.7,pitch=.55,zoom=1,center=[0,0,0],span=.3,drag=null,submitting=false;
  const canvas=el('retarget-canvas'),ctx=canvas.getContext('2d');
  function options(select,items,previous){
    select.replaceChildren();
    for(const job of items){const option=document.createElement('option');option.value=job.id;option.textContent=`${new Date(job.created_at*1000).toLocaleString()} · ${job.id.slice(0,8)} · ${job.status}`;select.append(option);}
    if(items.some(job=>job.id===previous))select.value=previous;
  }
  window.refreshRetargetJobs=items=>{
    jobs=items;
    options(el('retarget-source'),jobs.filter(j=>j.mode==='diffusion_video'&&j.status==='completed'),el('retarget-source').value);
    options(el('retarget-result'),jobs.filter(j=>j.mode==='generated_hand_retarget'),selected||el('retarget-result').value);
    el('retarget-start').disabled=submitting||!el('retarget-source').value;
    if(!selected&&el('retarget-result').value)watch(el('retarget-result').value);
  };
  async function watch(id){
    selected=id;clearTimeout(pollTimer);
    if([...el('retarget-result').options].some(option=>option.value===id))el('retarget-result').value=id;
    try{
      const job=await api('/api/jobs/'+id,{cache:'no-store'});if(selected!==id)return;
      const p=job.progress||{},active=['queued','running'].includes(job.status);
      el('retarget-status').textContent=`${id.slice(0,8)} · ${job.status}${p.total?` · ${p.step}/${p.total} frames`:''}${job.error?' · '+job.error:''}`;
      el('retarget-progress').hidden=!active;
      if(p.total){el('retarget-progress').max=p.total;el('retarget-progress').value=p.step||0;}else el('retarget-progress').removeAttribute('value');
      if(active)pollTimer=setTimeout(()=>watch(id),1600);
      if(job.status==='completed'&&loaded!==id){
        const preview=await api(job.artifacts['retarget_preview.json']);if(selected!==id)return;
        el('retarget-scene').hidden=true;el('retarget-scene').removeAttribute('src');el('retarget-scene-status').textContent='';
        data=preview;loaded=id;frame=0;playing=false;resetView();el('retarget-view').hidden=false;
        el('retarget-frame').max=data.frames.length-1;el('retarget-frame').value=0;
        for(const [link,name] of [['retarget-npz','retargeted_hand.npz'],['retarget-json','retarget_preview.json'],['retarget-manifest','manifest.json']])el(link).href=job.artifacts[name];
        el('retarget-demo').hidden=!job.artifacts['robot_demonstration.npz'];
        if(job.artifacts['robot_demonstration.npz'])el('retarget-demo').href=job.artifacts['robot_demonstration.npz'];
        draw();
      }
    }catch(error){if(selected!==id)return;el('retarget-status').textContent='Status update failed: '+error.message+' · retrying';pollTimer=setTimeout(()=>watch(id),3000);}
  }
  window.selectRetargetJob=id=>{if(id!==selected){playing=false;el('retarget-view').hidden=true;}watch(id);el('retarget-panel').scrollIntoView({behavior:'smooth',block:'start'});};
  el('retarget-scene-open').onclick=async()=>{
    if(!loaded)return;
    const id=loaded;el('retarget-scene-open').disabled=true;
    el('retarget-scene-status').textContent='Loading URDF meshes and calibrated scene…';
    try{
      const scene=await api(`/api/retarget/${id}/scene`,{method:'POST',headers:{'X-Retarget-Client':'1'}});
      if(loaded!==id)return;
      el('retarget-scene').src=scene.url;el('retarget-scene').hidden=false;
      el('retarget-scene-status').textContent='URDF hand + saved calibration-board pose. Table plane is estimated from the board; displayed table edges are illustrative. Orbit, play and adjust display dimensions inside the viewer.';
    }catch(error){el('retarget-scene-status').textContent=error.message;}
    finally{el('retarget-scene-open').disabled=false;}
  };
  el('retarget-start').onclick=async()=>{
    submitting=true;el('retarget-start').disabled=true;el('retarget-status').textContent='Submitting offline retargeting…';
    try{
      const job=await api('/api/retarget/generate',{method:'POST',headers:{'Content-Type':'application/json','X-Retarget-Client':'1'},body:JSON.stringify({source_job_id:el('retarget-source').value})});
      selected=job.id;data=null;loaded=null;playing=false;el('retarget-view').hidden=true;
      await history();await watch(job.id);
    }catch(error){el('retarget-status').textContent=error.message;}
    finally{submitting=false;el('retarget-start').disabled=!el('retarget-source').value;}
  };
  el('retarget-result').onchange=()=>{playing=false;el('retarget-view').hidden=true;loaded=null;watch(el('retarget-result').value);};
  function resetView(){
    yaw=.7;pitch=.55;zoom=1;
    if(!data)return;
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(const f of data.frames)for(const p of [...f.robot_points,...f.mano_points])for(let a=0;a<3;a++){lo[a]=Math.min(lo[a],p[a]);hi[a]=Math.max(hi[a],p[a]);}
    center=lo.map((v,a)=>(v+hi[a])/2);span=Math.max(.12,Math.hypot(...lo.map((v,a)=>hi[a]-v)))*1.15;
  }
  function rotate(p){
    const x=Math.cos(yaw)*p[0]-Math.sin(yaw)*p[1],y=Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1];
    return [x,Math.cos(pitch)*p[2]-Math.sin(pitch)*y,Math.cos(pitch)*y+Math.sin(pitch)*p[2]];
  }
  function project(p){const q=rotate(p.map((v,a)=>v-center[a])),scale=canvas.height/span*zoom;return [canvas.width/2+q[0]*scale,canvas.height/2-q[1]*scale];}
  function line(a,b,color,width,dashed=false){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[6,5]:[]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);}
  function draw(){
    if(!data)return;
    const f=data.frames[frame],colors=['#cd6c28','#247aca','#27975c','#9940b6','#bc9b16'];
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f1f3f5';ctx.fillRect(0,0,canvas.width,canvas.height);
    for(const [points,chains,dashed] of [[f.mano_points,data.mano_chains,true],[f.robot_points,data.robot_chains,false]]){
      if(dashed&&!el('retarget-show-mano').checked)continue;
      chains.forEach((chain,i)=>{
        for(let j=1;j<chain.length;j++)line(project(points[chain[j-1]]),project(points[chain[j]]),colors[i],dashed?1.5:3,dashed);
        if(!dashed)for(const index of chain){const p=project(points[index]);ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(...p,3.5,0,Math.PI*2);ctx.fill();}
      });
    }
    const axes=[[1,0,0],[0,1,0],[0,0,1]],axisColors=['#d54343','#23994c','#3376d1'];
    axes.forEach((axis,i)=>{
      const a=rotate(axis);line([72,85],[72+a[0]*45,85-a[1]*45],axisColors[i],2);
      ctx.fillStyle=axisColors[i];ctx.font='15px sans-serif';ctx.fillText('XYZ'[i],72+a[0]*58,85-a[1]*58);
      line(project([0,0,0]),project(axis.map(v=>v*.1)),axisColors[i],1);
    });
    ctx.fillStyle='#333';ctx.font='14px sans-serif';ctx.fillText('Robot base axes · metres',16,155);
    const p=f.robot_points[0];
    el('retarget-frame-info').textContent=`Frame ${frame+1}/${data.frames.length} · ${(frame/data.fps).toFixed(2)} s · Palm XYZ: ${p.map(v=>v.toFixed(3)).join(', ')} m · Mean scaled fingertip fit error: ${(1000*f.mean_fingertip_error_m).toFixed(1)} mm`;
    el('retarget-joints').textContent=(f.arm_q_rad?`Arm [rad]: ${f.arm_q_rad.map(v=>v.toFixed(3)).join(', ')} · Palm IK error: ${(1000*f.arm_position_error_m).toFixed(2)} mm\n`:'')+Array.from({length:5},(_,finger)=>`${['Thumb','Index','Middle','Ring','Pinky'][finger]} [rad]: ${f.hand_q_rad.slice(finger*4,finger*4+4).map(v=>v.toFixed(3)).join(', ')}`).join('\n');
    el('retarget-play').textContent=playing?'Pause':'Play';el('retarget-frame').value=frame;
  }
  el('retarget-play').onclick=()=>{playing=!playing;lastTime=performance.now();draw();};
  el('retarget-frame').oninput=()=>{playing=false;frame=Number(el('retarget-frame').value);draw();};
  el('retarget-reset').onclick=()=>{resetView();draw();};el('retarget-show-mano').onchange=draw;
  canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
  canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
  canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw();};
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
  function animate(time){if(playing&&data){const steps=Math.floor((time-lastTime)*data.fps/1000);if(steps){frame=(frame+steps)%data.frames.length;lastTime+=steps*1000/data.fps;draw();}}requestAnimationFrame(animate);}
  requestAnimationFrame(animate);window.refreshRetargetJobs(lastJobs);
})();

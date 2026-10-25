(()=>{
  const el=id=>document.getElementById(id),canvas=el('simulation-pick'),ctx=canvas.getContext('2d');
  let jobs=[],sourceId=null,image=null,pixel=null,selected=null,loaded=null,timer=null,busy=false;
  function fill(id,items,previous){const select=el(id);select.replaceChildren();for(const job of items){const option=document.createElement('option');option.value=job.id;option.textContent=`${new Date(job.created_at*1000).toLocaleString()} · ${job.id.slice(0,8)} · ${job.status}`;select.append(option);}if(items.some(j=>j.id===previous))select.value=previous;}
  window.refreshSimulationJobs=items=>{
    jobs=items;
    fill('simulation-source',jobs.filter(j=>j.mode==='generated_hand_retarget'&&j.status==='completed'),el('simulation-source').value);
    fill('simulation-result',jobs.filter(j=>j.mode==='mujoco_replay'),selected||el('simulation-result').value);
    if(el('simulation-source').value!==sourceId)loadSource();
    if(!selected&&el('simulation-result').value)watch(el('simulation-result').value);
  };
  async function loadSource(){
    const id=el('simulation-source').value;sourceId=id;image=null;pixel=null;canvas.hidden=true;el('simulation-start').disabled=true;
    if(!id)return;
    el('simulation-selection').textContent='Loading the paired first frame…';
    try{
      const retarget=jobs.find(j=>j.id===id),generation=await api('/api/jobs/'+retarget.source_job_id);
      if(!generation.input_url)throw new Error('The paired first frame is unavailable.');
      const photo=await imageFromURL(generation.input_url);if(id!==sourceId)return;
      image=photo;canvas.width=photo.naturalWidth;canvas.height=photo.naturalHeight;canvas.hidden=false;draw();
      el('simulation-selection').textContent='Click the center of the cube’s top face, then check its size, mass and friction.';
    }catch(error){if(id===sourceId)el('simulation-selection').textContent=error.message;}
  }
  function draw(){if(!image)return;ctx.drawImage(image,0,0,canvas.width,canvas.height);if(pixel){const [x,y]=pixel;ctx.strokeStyle='#ff2b2b';ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,10,0,Math.PI*2);ctx.moveTo(x-18,y);ctx.lineTo(x+18,y);ctx.moveTo(x,y-18);ctx.lineTo(x,y+18);ctx.stroke();}}
  canvas.onclick=event=>{if(!image)return;const r=canvas.getBoundingClientRect();pixel=[(event.clientX-r.left)*canvas.width/r.width,(event.clientY-r.top)*canvas.height/r.height];draw();el('simulation-selection').textContent=`Selected pixel (${pixel.map(v=>v.toFixed(1)).join(', ')}). Placement assumes the cube rests on the calibrated table.`;el('simulation-start').disabled=busy;};
  el('simulation-source').onchange=loadSource;
  async function watch(id){
    selected=id;clearTimeout(timer);
    if([...el('simulation-result').options].some(o=>o.value===id))el('simulation-result').value=id;
    try{
      const job=await api('/api/jobs/'+id,{cache:'no-store'});if(selected!==id)return;
      const p=job.progress||{},active=['queued','running'].includes(job.status);
      el('simulation-status').textContent=`${id.slice(0,8)} · ${job.status} · ${p.stage||'queued'}${p.total?` · ${p.step}/${p.total} frames`:''}${job.error?' · '+job.error:''}`;
      el('simulation-progress').hidden=!active;
      if(p.total){el('simulation-progress').max=p.total;el('simulation-progress').value=p.step||0;}else el('simulation-progress').removeAttribute('value');
      if(active)timer=setTimeout(()=>watch(id),1600);
      if(job.status==='completed'&&loaded!==id){
        loaded=id;el('simulation-output').hidden=false;
        const files=job.artifacts||{};
        el('simulation-camera-video').src=files['mujoco_camera.mp4'];el('simulation-overview-video').src=files['mujoco_overview.mp4'];
        el('simulation-scene-download').href=files['mujoco_scene.zip'];el('simulation-motion-download').href=files['mujoco_motion.npz'];el('simulation-report-download').href=files['manifest.json'];
        const m=await api(files['manifest.json']);if(selected!==id)return;
        el('simulation-metrics').textContent=`${m.lift_observed?'Lift observed':'No sustained lift observed'} in simulation · ${m.hand_contact_frames} hand-contact frames${Number.isFinite(m.minimum_hand_cube_surface_gap_m)?' · closest hand–cube gap '+(m.minimum_hand_cube_surface_gap_m*1000).toFixed(1)+' mm':''} · maximum cube lift ${((m.max_cube_lift_m||0)*1000).toFixed(1)} mm · finger tracking RMS ${(m.joint_tracking_rms_rad||0).toFixed(3)} rad. Results depend on the estimated cube and contact properties.`;
      }
    }catch(error){if(selected!==id)return;el('simulation-status').textContent='Status update failed: '+error.message;timer=setTimeout(()=>watch(id),3000);}
  }
  function select(id){for(const video of ['simulation-camera-video','simulation-overview-video'])el(video).pause();el('simulation-output').hidden=true;loaded=null;watch(id);}
  window.selectSimulationJob=id=>{select(id);el('simulation-panel').scrollIntoView({behavior:'smooth'});};
  el('simulation-result').onchange=()=>select(el('simulation-result').value);
  el('simulation-start').onclick=async()=>{
    if(!pixel||!sourceId||busy)return;
    busy=true;el('simulation-start').disabled=true;el('simulation-status').textContent='Submitting MuJoCo replay…';
    try{
      const job=await api('/api/simulation/generate',{method:'POST',headers:{'Content-Type':'application/json','X-Retarget-Client':'1'},body:JSON.stringify({source_job_id:sourceId,pixel,edge_m:Number(el('simulation-edge').value)/1000,mass_kg:Number(el('simulation-mass').value)/1000,yaw_deg:Number(el('simulation-yaw').value),friction:Number(el('simulation-friction').value)})});
      selected=job.id;await history();select(job.id);
    }catch(error){el('simulation-status').textContent=error.message;}
    finally{busy=false;el('simulation-start').disabled=!pixel;}
  };
  window.refreshSimulationJobs(lastJobs);
})();

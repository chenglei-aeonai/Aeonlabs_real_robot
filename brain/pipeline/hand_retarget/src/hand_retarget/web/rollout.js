(()=>{
  const el=id=>document.getElementById(id);let selected=null,timer=null,ready=false,hardware=null,runId=null,runPending=false,preparing=false;
  function fill(id,jobs,previous){const select=el(id);select.replaceChildren();for(const job of jobs){const option=document.createElement('option');option.value=job.id;option.textContent=`${new Date(job.created_at*1000).toLocaleString()} · ${job.id.slice(0,8)} · ${job.status}`;select.append(option);}if(jobs.some(j=>j.id===previous))select.value=previous;}
  window.refreshRolloutJobs=jobs=>{
    fill('rollout-source',jobs.filter(j=>j.mode==='generated_hand_retarget'&&j.status==='completed'),el('rollout-source').value);
    fill('rollout-result',jobs.filter(j=>j.mode==='robot_rollout'),selected||el('rollout-result').value);
    el('rollout-prepare').disabled=preparing||!el('rollout-source').value;
    if(!selected&&el('rollout-result').value)watch(el('rollout-result').value);
  };
  async function watch(id){
    if(selected!==id){ready=false;updateHardwareButtons();}
    selected=id;clearTimeout(timer);
    try{
      const job=await api('/api/jobs/'+id,{cache:'no-store',signal:AbortSignal.timeout(10000)});if(selected!==id)return;
      const resultSelect=el('rollout-result');
      let option=[...resultSelect.options].find(item=>item.value===id);
      if(!option){option=document.createElement('option');option.value=id;resultSelect.append(option);}
      option.textContent=`${new Date(job.created_at*1000).toLocaleString()} · ${id.slice(0,8)} · ${job.status}`;resultSelect.value=id;
      const cached=lastJobs.findIndex(item=>item.id===id);if(cached>=0)lastJobs[cached]=job;
      el('rollout-status').textContent=`${job.status} · ${job.progress?.stage||'queued'}${job.error?' · '+job.error:''}`;
      const active=['queued','running'].includes(job.status);preparing=active;el('rollout-prepare').disabled=active;el('rollout-prepare').textContent=active?'Preparing rollout…':'Prepare and check robot rollout';el('rollout-progress').hidden=!active;
      if(job.progress?.total){el('rollout-progress').max=job.progress.total;el('rollout-progress').value=job.progress.step||0;}else el('rollout-progress').removeAttribute('value');
      el('rollout-output').hidden=job.status!=='completed';
      ready=job.status==='completed'&&job.rollout?.execution_ready;updateHardwareButtons();
      if(active)timer=setTimeout(()=>watch(id),1600);
      if(job.status==='completed'){
        const r=job.rollout;
        const timing=r.timing_policy==='local_time_allocation'
          ? `Local timing · original ${r.source_duration_s.toFixed(2)} s · motion ${r.duration_s.toFixed(2)} s · up to ${(r.native_timing?.maximum_settle_time_s ?? 1.5).toFixed(2)} s settling · 8 s budget (rechecked from live pose)`
          : r.timing_policy==='source_rate_demo'
          ? `Original ${r.source_duration_s.toFixed(2)} s · requested ${r.speed_scale}× · checked ${r.effective_speed_scale}× · ${r.duration_s.toFixed(2)} s (rechecked from live pose)`
          : `Historical conservative timing · ${Math.round((r.speed_scale??1)*100)}% of prepared baseline · ${(r.live_relative_duration_s??r.duration_s).toFixed(1)} s`;
        el('rollout-metrics').textContent=`Offline checks passed · ${timing}${r.start_policy==='live_cartesian_v2'?' · Cartesian path anchored to live pose':r.start_policy==='live_relative_v1'?' · joint-relative legacy start':''} · minimum sampled table clearance ${(r.minimum_table_clearance_m*1000).toFixed(1)} mm · maximum arm IK position error ${(r.maximum_ik_position_error_m*1000).toFixed(2)} mm. No hardware motion has been sent.`;
        el('rollout-check-command').textContent=r.check_command;el('rollout-send-command').textContent=r.send_command;
        el('rollout-download').href=job.artifacts['robot_rollout.npz'];el('rollout-report').href=job.artifacts['manifest.json'];
      }
    }catch(error){if(selected!==id)return;el('rollout-status').textContent='Status update interrupted; retrying… '+error.message;timer=setTimeout(()=>watch(id),2000);}
  }
  window.selectRolloutJob=id=>{watch(id);el('rollout-panel').scrollIntoView({behavior:'smooth'});};
  el('rollout-result').onchange=()=>watch(el('rollout-result').value);
  el('rollout-speed').onchange=()=>{
    ready=false;updateHardwareButtons();
    el('rollout-status').textContent='Speed changed. Prepare again to apply it; the selected rollout retains its saved timing.';
  };
  el('rollout-prepare').onclick=async()=>{
    if(preparing)return;preparing=true;ready=false;updateHardwareButtons();
    el('rollout-prepare').disabled=true;el('rollout-status').textContent='Submitting offline rollout preparation…';
    try{const job=await api('/api/rollout/prepare',{method:'POST',headers:{'Content-Type':'application/json','X-Retarget-Client':'1'},body:JSON.stringify({source_job_id:el('rollout-source').value,speed_scale:Number(el('rollout-speed').value)})});selected=job.id;watch(job.id);history().catch(()=>{});}
    catch(error){preparing=false;el('rollout-prepare').disabled=!el('rollout-source').value;el('rollout-status').textContent=error.message;}
  };
  function updateHardwareButtons(){
    el('robot-run').disabled=!ready||runPending||!!hardware?.active;
    el('robot-run-help').textContent=hardware?.active?'Physical replay active. Stop robot interrupts this replay.':ready?'Ready: click Run on robot, then confirm SEND.':preparing?'Preparing the trajectory. Run on robot enables when checks finish.':'Select a completed rollout below, or prepare one first.';
    el('robot-stop').disabled=!hardware?.active;
  }
  async function pollHardware(){
    try{
      hardware=await api('/api/robot/status',{cache:'no-store',signal:AbortSignal.timeout(1500)});
      if(hardware.active&&runId===hardware.run_id)await api('/api/robot/heartbeat',{method:'POST',headers:{'Content-Type':'application/json','X-Retarget-Client':'1'},body:JSON.stringify({run_id:runId}),signal:AbortSignal.timeout(1500)});
      el('robot-state').textContent=`Robot replay: ${hardware.state}${hardware.job_id?' · rollout '+hardware.job_id.slice(0,8):''}${hardware.stop_reason?' · '+hardware.stop_reason:''}`+(hardware.state==='stop_unconfirmed'?' · Use the physical e-stop; a stopped state could not be confirmed.':'');
      el('robot-log').hidden=!hardware.log;el('robot-log').textContent=hardware.log||'';updateHardwareButtons();
    }catch(error){el('robot-state').textContent='Robot status unavailable. The browser lease will interrupt this replay. Use the physical e-stop if motion continues. '+error.message;}
    setTimeout(pollHardware,750);
  }
  el('robot-run').onclick=()=>{
    el('robot-confirm-text').value='';el('robot-workspace-clear').checked=false;el('robot-confirm-error').hidden=true;
    el('robot-confirm-motion').textContent=(el('robot-execution-mode').value==='arm-only'?'Only the arm will be commanded. Hand feedback remains required for table checks. ':'The arm and fingers will move. ')+'Keep the workspace clear and the e-stop accessible. Stay on this page; loss of the browser heartbeat stops this replay.';
    el('robot-confirm-dialog').showModal();el('robot-confirm-text').focus();
  };
  el('robot-confirm-cancel').onclick=()=>el('robot-confirm-dialog').close();
  el('robot-confirm-start').onclick=async()=>{
    if(runPending)return;
    runPending=true;el('robot-confirm-start').disabled=true;updateHardwareButtons();
    try{
      hardware=await api('/api/robot/run',{method:'POST',headers:{'Content-Type':'application/json','X-Retarget-Client':'1'},body:JSON.stringify({source_job_id:selected,confirmation:el('robot-confirm-text').value,workspace_clear:el('robot-workspace-clear').checked,arm_only:el('robot-execution-mode').value==='arm-only'})});
      runId=hardware.run_id;el('robot-confirm-dialog').close();el('robot-state').textContent='Starting physical robot replay…';updateHardwareButtons();
    }catch(error){el('robot-confirm-error').hidden=false;el('robot-confirm-error').textContent=error.message;}
    finally{runPending=false;el('robot-confirm-start').disabled=false;updateHardwareButtons();}
  };
  el('robot-stop').onclick=async()=>{
    el('robot-state').textContent='Requesting robot stop…';
    try{hardware=await api('/api/robot/stop',{method:'POST',headers:{'X-Retarget-Client':'1'}});updateHardwareButtons();}
    catch(error){el('robot-state').textContent='Stop request failed. Use the physical e-stop. '+error.message;}
  };
  pollHardware();
  window.refreshRolloutJobs(lastJobs);
})();

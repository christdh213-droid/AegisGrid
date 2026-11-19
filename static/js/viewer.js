import {Transform,GRID} from './transform.js';
import {decodeFrame} from './protocol.js';
import {readLocalScan} from './localScan.js';
const $=id=>document.getElementById(id),canvas=$('rightCanvas'),ctx=canvas.getContext('2d');
const raster=$('leftCanvas'),rctx=raster.getContext('2d'),transform=new Transform();
let latest=null,socket=null,retry=500,polling=false,pollTimer=null,wipe=.5,drag=null,playing=false;
let mode = 'split', hudHidden = false, lowFpsSince = null;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
function mapWidth(){ return innerWidth - (!hudHidden && innerWidth > 820 ? 366 : 0); }
function toggleHud(){ hudHidden=!hudHidden; $('telemetry').hidden=hudHidden; $('hideHud').textContent=hudHidden?'Show HUD':'Hide HUD'; $('hideHud').setAttribute('aria-expanded',String(!hudHidden)); document.body.classList.toggle('hud-hidden',hudHidden); }
$('hideHud').onclick=toggleHud;
let localScan = false;
const localRaster = document.createElement('canvas');
localRaster.width = localRaster.height = GRID.dim;
let drawCount=0,received=0,lastStats=performance.now(),latency=null,receivedHz=null;
const trackLabels = new Map();
const labelCounts = {VEH:0,PED:0};
let lastFrameId = -1;
window.viewerDiagnostics={renderFps:null,receivedFps:null,latencyMs:null,framesReceived:0,errors:[]};
const palette=['#cf9269','#74b4ac','#4f9eab','#265779','#152c51'];
const riskPalette = ['#378c75','#c79851','#b65158','#344759'];
const lut=Array.from({length:256},(_,i)=>i===0?[6,8,16]:[Math.round(25+156*i/255),Math.round(10+35*i/255),Math.round(22+35*i/255)]);
function resize(){const dpr=devicePixelRatio||1;canvas.width=Math.round(innerWidth*dpr);canvas.height=Math.round(innerHeight*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);transform.resize(mapWidth(),innerHeight);}
addEventListener('resize',resize);resize();
function showError(error){$('error').textContent=String(error);window.viewerDiagnostics.errors.push(String(error));}
function updateConnection(text, live = false) {
  $('connection').textContent = text;
  $('connection').toggleAttribute('data-live', live);
}
function setMode(value) {
  mode = value;
  $('riskLegend').hidden = mode !== 'risk';
  document.querySelector('.modes').style.setProperty('--active', ['split','elevation','quadtree','risk'].indexOf(mode));
  $('wipe').hidden = mode !== 'split' || localScan;
  $('baselineLabel').hidden = !localScan && !mode === 'split';
  $('adaptiveLabel').hidden = localScan || false;
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.setAttribute('aria-pressed', String(!localScan && button.dataset.mode === mode));
  });
}
document.querySelectorAll('[data-mode]').forEach(button => {
  button.onclick = () => { closeLocalScan(); setMode(button.dataset.mode); };
});
function closeLocalScan() {
  localScan = false;
  $('returnLive').hidden = true;
  $('baselineLabel').textContent = 'Baseline · 0.25 m grid';
  setMode(mode);
}
$('returnLive').onclick = closeLocalScan;
$('upload').onclick = () => $('file').click();
$('file').onchange = async event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  try {
    const scan = await readLocalScan(file);
    const context = localRaster.getContext('2d');
    const image = context.createImageData(GRID.dim, GRID.dim);
    for (let row = 0; row < GRID.dim; row++) for (let col = 0; col < GRID.dim; col++) {
      const z = scan.cells[row * GRID.dim + col];
      const i = transform.rasterIndex(col, row);
      const color = Number.isFinite(z) ? lut[1 + Math.round(254 * Math.max(0, Math.min(1, (z + 2) / 6)))] : lut[0];
      image.data.set([...color, 255], i);
    }
    context.putImageData(image, 0, 0);
    localScan = true;
    $('baselineLabel').textContent = `Local scan · ${scan.count.toLocaleString()} points in view`;
    $('returnLive').hidden = false;
    $('error').textContent = '';
    transform.reset();
    setMode(mode);
  } catch (error) { showError(error.message); }
};
function displayTracks(tracks) {
  return (tracks ?? []).filter(track => track.age >= 3 && track.speed >= 0.4).flatMap(track => {
    const [x, y] = track.centroid ?? [track.x, track.y];
    if (!Number.isFinite(x) || !Number.isFinite(y)) return [];
    const kind = ['PED','PEDESTRIAN'].includes(track.class ?? track.classification) ? 'PED' : 'VEH';
    let label = track.label;
    if (!label && track.track_id != null) {
      if (!trackLabels.has(track.track_id)) {
        trackLabels.set(track.track_id, `${kind}-${String(++labelCounts[kind]).padStart(2,'0')}`);
      }
      label = trackLabels.get(track.track_id);
    }
    return [{x,y,label,speed:track.speed}];
  });
}
function receive(buffer){
  try {
    latest=decodeFrame(buffer);received++;window.viewerDiagnostics.framesReceived++;
    if (latest.frameId < lastFrameId) {
      trackLabels.clear();
      labelCounts.VEH = labelCounts.PED = 0;
    }
    lastFrameId = latest.frameId;
    latest.displayTracks = displayTracks(latest.meta.dynamic_tracks);
    latency=Date.now()-latest.serverTime;window.viewerDiagnostics.latencyMs=latency;
    const image=rctx.createImageData(GRID.dim,GRID.dim);
    for(let row=0;row<GRID.dim;row++)for(let col=0;col<GRID.dim;col++){
      const color=lut[latest.raster[row*GRID.dim+col]],i=transform.rasterIndex(col,row);
      image.data[i]=color[0];image.data[i+1]=color[1];image.data[i+2]=color[2];image.data[i+3]=255;
    }
    rctx.putImageData(image,0,0);
    $('adaptiveLabel').textContent=`Adaptive Quadtree · ${latest.n.toLocaleString()} cells`;
    const m=latest.meta.metrics,e=latest.meta.extras;
    $('reduction').textContent = `${m.cell_reduction_pct.toFixed(1)}%`;
    $('memory').textContent = `${m.memory_saved_pct.toFixed(1)}%`;
    $('metrics').textContent = `Frame ${latest.frameId} · RMSE ${m.obstacle_rmse_m.toFixed(4)} m · Ghosts ${m.ghost_cells_detected} · Transport ${latency.toFixed(1)} ms`;
    if(e.runtime_config) syncControls(e.runtime_config);
    $('routeStatus').textContent = localScan ? 'Return to live to set a destination.' :
      (e.path_status ? `${e.path_status}${e.path_reason ? ' · ' + e.path_reason : ''} · Click map to set goal` : 'Route planner is off');
    $('routeCompare').hidden = e.fixed_plan_ms == null;
    if (e.fixed_plan_ms != null) {
      const savings = e.nodes_saved_pct == null ? '' : ` · ${e.nodes_saved_pct.toFixed(1)}% fewer nodes`;
      $('routeCompare').textContent = `Fixed sample: ${e.fixed_grid_nodes_expanded} nodes · ${e.fixed_plan_ms.toFixed(1)} ms · ${e.fixed_sample_age_s.toFixed(1)} s old${savings}`;
    }
    if (e.governor_mode) {
      const scale = e.governor_mode === 'Unrestricted' ? 'Host compute' : 'Emulated edge profile (host-scaled)';
      const warning = e.governor_state === 'BUDGET_LOCKED' ? ' · Budget unmet at threshold ceilings' : '';
      $('governorReadout').textContent = `${scale} · ${e.governor_state} · ${e.governed_latency_ms.toFixed(1)} / ${e.target_budget_ms} ms · Height / gradient ${e.active_merge_threshold.toFixed(2)} / ${e.active_gradient_threshold.toFixed(2)} m${warning}`;
      $('radioRate').hidden = e.radio_mb_s_fixed == null || e.radio_mb_s_adaptive == null;
      if (!$('radioRate').hidden) {
        $('radioRate').textContent = `Estimated map radio · Fixed ${e.radio_mb_s_fixed.toFixed(2)} / adaptive ${e.radio_mb_s_adaptive.toFixed(2)} MB/s`;
      }
    }
    if(e.safe_pct != null){
      $('perception').textContent = `Filtered ${e.filtered_noise_points} points · IMU ${e.imu_pitch_deg.toFixed(1)}° / ${e.imu_roll_deg.toFixed(1)}°${e.tilt_corrected ? ' (levelled)' : ''}`;
      ['safe','caution','hazard','unknown'].forEach((key, i) => {
        const share = e[`${key}_pct`];
        $(key).textContent = `${share.toFixed(1)}%`;
        $('riskBar').children[i].style.flex = String(share);
      });
    }
  }catch(error){showError(error);}
}
async function poll(){
  if(!polling)return;
  try {
    const response=await fetch('/api/frame',{headers:{Accept:'application/octet-stream'},cache:'no-store'});
    if(!response.ok)throw Error('REST '+response.status);
    const buffer=await response.arrayBuffer();
    if (!polling || socket?.readyState === WebSocket.OPEN) return;
    receive(buffer);
    updateConnection('REST polling · Reconnecting');
  } catch(error) {
    if (polling) updateConnection('Reconnecting');
  }
  if (polling) pollTimer=setTimeout(poll,100);
}
function fallback(){if(!polling){polling=true;poll();}}
function connect(){
  $('connection').textContent='Connecting';
  try{socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/stream`);}catch(error){$('connection').textContent='Reconnecting';fallback();setTimeout(connect,retry);retry=Math.min(10000,retry*2);return;}socket.binaryType='arraybuffer';
  socket.onopen=()=>{retry=500;polling=false;clearTimeout(pollTimer);updateConnection('LIVE', true);};
  socket.onmessage=event=>{
    if(event.data instanceof ArrayBuffer){receive(event.data);return;}
    try{const message=JSON.parse(event.data);if(message.type==='error')showError(message.message);else if(message.type==='control_ack'){playing=message.playing;if(message.config)syncControls(message.config);}}catch(error){showError(error);}
  };
  socket.onerror=()=>socket.close();
  socket.onclose=()=>{updateConnection('Reconnecting');fallback();setTimeout(connect,retry);retry=Math.min(10000,retry*2);};
}
async function command(message){
  if(socket?.readyState===1){socket.send(JSON.stringify(message));return;}
  try{const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(message)});const result=await response.json();if(!response.ok)throw Error(result.detail);playing=result.playing;if(result.config)syncControls(result.config);if(['step','reset','set_goal'].includes(message.action)){const frame=await fetch('/api/frame',{headers:{Accept:'application/octet-stream'}});receive(await frame.arrayBuffer());}}catch(error){showError(error);}
}
for(const action of ['play','pause','step','reset'])$(action).onclick=()=>{if(action==='reset'){transform.reset();wipe=.5;$('wipe').style.left='50%';}command({action});};
$('home').onclick=()=>transform.reset();$('speed').onchange=event=>command({action:'configure',settings:{speed:Number(event.target.value)}});
const configToggles = [...document.querySelectorAll('[data-config]')];
function syncControls(config) {
  $('speed').value = String(config.speed);
  $('governorProfile').value = config.governor_profile ?? 'Unrestricted';
  for (const toggle of configToggles) toggle.checked = Boolean(config[toggle.dataset.config]);
  if (document.activeElement !== $('dust')) {
    $('dust').value = String(Math.round(config.outlier_ratio * 100));
    $('dustValue').textContent = `${$('dust').value}%`;
  }
}
$('governorProfile').onchange = event => command({action:'set_profile',profile:event.target.value});
for (const toggle of configToggles) {
  toggle.onchange = () => command({action:'configure',settings:{[toggle.dataset.config]:toggle.checked}});
}
$('dust').oninput = () => { $('dustValue').textContent = `${$('dust').value}%`; };
$('dust').onchange = () => command({action:'configure',settings:{outlier_ratio:Number($('dust').value)/100}});
canvas.onpointerdown=event=>{
  if (event.button !== 0) return;
  drag={x:event.clientX,y:event.clientY,moved:false};
  canvas.setPointerCapture(event.pointerId);
};
canvas.onpointermove=event=>{
  if (!drag) return;
  if (!drag.moved && Math.hypot(event.clientX-drag.x,event.clientY-drag.y) <= 5) return;
  transform.pan(event.clientX-drag.x,event.clientY-drag.y);
  drag={x:event.clientX,y:event.clientY,moved:true};
};
canvas.onpointerup=event=>{
  if (drag && !drag.moved && !localScan && latest?.meta.extras.runtime_config?.planner_enabled) {
    const {x,y} = transform.screenToWorld(event.clientX,event.clientY);
    command({action:'set_goal',goal_xy:[x,y]});
  }
  drag=null;
};
canvas.onpointercancel=()=>drag=null;
canvas.onwheel=event=>{event.preventDefault();transform.zoomAt(Math.exp(-event.deltaY*.001),event.clientX,event.clientY);};
canvas.ondblclick=()=>transform.reset();
const handle=$('wipe');let wiping=false;
handle.onpointerdown=event=>{wiping=true;handle.setPointerCapture(event.pointerId);};
handle.onpointermove=event=>{if(wiping){wipe=Math.max(.02,Math.min(.98,event.clientX/innerWidth));handle.style.left=`${wipe*100}%`;}};
handle.onpointerup=handle.onpointercancel=()=>wiping=false;
handle.onkeydown=event=>{if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();wipe=Math.max(.02,Math.min(.98,wipe+(event.key==='ArrowLeft'?-.02:.02)));handle.style.left=`${wipe*100}%`;}};
addEventListener('keydown',event=>{if(event.target.matches('button,select,input'))return;if(event.key==='0')transform.reset();if(event.code==='Space'){event.preventDefault();command({action:playing?'pause':'play'});}if(event.key==='ArrowRight')command({action:'step'});});
function draw(now){
  const width=innerWidth,height=innerHeight,divider=width*wipe;
  const targetWidth=mapWidth(); transform.resize(reducedMotion.matches?targetWidth:transform.width+(targetWidth-transform.width)*.12,height);
  ctx.fillStyle='#050811';ctx.fillRect(0,0,width,height);
  if(localScan) {
    const extent = transform.extentRect();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(localRaster,extent.x,extent.y,extent.width,extent.height);
  } else if(latest){
    const extent=transform.extentRect();
    if (mode === 'split' || false) {
    ctx.save();ctx.beginPath();ctx.rect(0,0,mode === 'split' ? divider : width,height);ctx.clip();ctx.imageSmoothingEnabled=false;ctx.drawImage(raster,extent.x,extent.y,extent.width,extent.height);
    if(transform.pixelsPerCell>=6){ctx.strokeStyle='#49303a';ctx.lineWidth=.5;ctx.beginPath();for(let i=0;i<=GRID.dim;i++){const a=transform.gridToScreen(i,0),b=transform.gridToScreen(i,GRID.dim),c=transform.gridToScreen(0,i),d=transform.gridToScreen(GRID.dim,i);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.moveTo(c.x,c.y);ctx.lineTo(d.x,d.y);}ctx.stroke();}ctx.restore();
    }
    if (true) {
    const left = mode === 'split' ? divider : 0;
    ctx.save();ctx.beginPath();ctx.rect(left,0,width-left,height);ctx.clip();
    for(let i=0;i<latest.n;i++){const r=transform.cellRect(latest.x[i],latest.y[i],latest.level[i]);ctx.fillStyle=mode === 'risk' ? riskPalette[(latest.flags[i] >> 1) & 3] : mode === 'elevation' ? `rgb(${lut[Math.max(1,Math.min(255,Math.round((latest.zMax[i]/100+2)*254/8)+1))].join(',')})` : palette[latest.level[i]];ctx.fillRect(r.x,r.y,r.width,r.height);if(r.width>=4){ctx.strokeStyle='rgba(148,197,255,0.18)';ctx.lineWidth=.4;ctx.strokeRect(r.x,r.y,r.width,r.height);}}ctx.restore();
    }
    // World overlays intentionally sit outside both wipe clips.
    const route = latest.meta.extras;
    ctx.save();
    if (route.planned_path?.length) {
      ctx.beginPath();
      route.planned_path.forEach(([x,y],i)=>{
        const point=transform.worldToScreen(x,y);
        if(i)ctx.lineTo(point.x,point.y);else ctx.moveTo(point.x,point.y);
      });
      const blocked = route.path_status === 'BLOCKED';
      ctx.strokeStyle = blocked ? '#b6ac9590' : '#6ef5dc';
      ctx.lineWidth = blocked ? 2 : 4;
      if(blocked)ctx.setLineDash([7,8]);
      ctx.stroke();
    }
    if (route.goal_xy) {
      const goal=transform.worldToScreen(...route.goal_xy);
      ctx.setLineDash([]);
      ctx.strokeStyle = route.path_status === 'BLOCKED' ? '#ffc36e' : '#6ef5dc';
      ctx.lineWidth=2;ctx.beginPath();ctx.arc(goal.x,goal.y,7,0,Math.PI*2);ctx.stroke();
    }
    ctx.restore();
    const occupiedTags = [];
    ctx.font = '11px ui-monospace,Consolas,monospace';
    for (const track of latest.displayTracks) {
      const point = transform.worldToScreen(track.x, track.y);
      ctx.strokeStyle = '#9de1bc';
      ctx.lineWidth = 1;
      ctx.strokeRect(point.x-4,point.y-4,8,8);
      if (!track.label) continue;
      const text = `${track.label} · ${track.speed.toFixed(1)} m/s`;
      const tagWidth = ctx.measureText(text).width + 8;
      if (tagWidth > width) continue;
      const x = Math.max(0, Math.min(point.x + 7, width - tagWidth));
      for (const offset of [0, -16, 16, -32, 32]) {
        const y = point.y + offset - 11;
        if (y < 0 || y + 15 > height) continue;
        if (occupiedTags.some(tag => x < tag.right && x + tagWidth > tag.x && y < tag.y + 15 && y + 15 > tag.y)) continue;
        occupiedTags.push({x,y,right:x+tagWidth});
        ctx.fillStyle = '#050811dd';
        ctx.fillRect(x,y,tagWidth,15);
        ctx.fillStyle = '#e5e9ef';
        ctx.fillText(text,x+4,y+11);
        break;
      }
    }
    const ego=transform.worldToScreen(0,0);ctx.fillStyle='#c9e4ee';ctx.beginPath();ctx.arc(ego.x,ego.y,4,0,Math.PI*2);ctx.fill();
    if (mode === 'split') {
      ctx.strokeStyle='#9fdaed';ctx.beginPath();ctx.moveTo(divider,0);ctx.lineTo(divider,height);ctx.stroke();
    }
    drawCount++;
  }
  if(now-lastStats>=1000){const seconds=(now-lastStats)/1000;const fps=drawCount/seconds;receivedHz=received/seconds;$('fps').textContent=`Render ${fps.toFixed(1)} FPS · Received ${receivedHz.toFixed(1)} Hz`;Object.assign(window.viewerDiagnostics,{renderFps:fps,receivedFps:receivedHz}); if(socket?.readyState===1)updateConnection(`LIVE ${fps.toFixed(0)} FPS`,true); if(fps<20 && latest){lowFpsSince ??= now; if(now-lowFpsSince>=3000)document.body.classList.add('lite');}else lowFpsSince=null;drawCount=0;received=0;lastStats=now;}
  requestAnimationFrame(draw);
}
connect();setTimeout(()=>{if(socket?.readyState!==1)fallback();},1500);requestAnimationFrame(draw);

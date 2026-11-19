import {Transform,GRID} from './transform.js';
import {decodeFrame} from './protocol.js';
const $=id=>document.getElementById(id),canvas=$('rightCanvas'),ctx=canvas.getContext('2d');
const raster=$('leftCanvas'),rctx=raster.getContext('2d'),transform=new Transform();
let latest=null,socket=null,retry=500,polling=false,pollTimer=null,wipe=.5,drag=null,playing=false;
let drawCount=0,received=0,lastStats=performance.now(),latency=null,receivedHz=null;
window.viewerDiagnostics={renderFps:null,receivedFps:null,latencyMs:null,framesReceived:0,errors:[]};
const palette=['#cf9269','#74b4ac','#4f9eab','#265779','#152c51'];
const lut=Array.from({length:256},(_,i)=>i===0?[6,8,16]:[Math.round(25+156*i/255),Math.round(10+35*i/255),Math.round(22+35*i/255)]);
function resize(){const dpr=devicePixelRatio||1;canvas.width=Math.round(innerWidth*dpr);canvas.height=Math.round(innerHeight*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);transform.resize(innerWidth,innerHeight);}
addEventListener('resize',resize);resize();
function showError(error){$('error').textContent=String(error);window.viewerDiagnostics.errors.push(String(error));}
function receive(buffer){
  try {
    latest=decodeFrame(buffer);received++;window.viewerDiagnostics.framesReceived++;
    latency=Date.now()-latest.serverTime;window.viewerDiagnostics.latencyMs=latency;
    const image=rctx.createImageData(GRID.dim,GRID.dim);
    for(let row=0;row<GRID.dim;row++)for(let col=0;col<GRID.dim;col++){
      const color=lut[latest.raster[row*GRID.dim+col]],i=transform.rasterIndex(col,row);
      image.data[i]=color[0];image.data[i+1]=color[1];image.data[i+2]=color[2];image.data[i+3]=255;
    }
    rctx.putImageData(image,0,0);
    $('adaptiveLabel').textContent=`Adaptive Quadtree · ${latest.n.toLocaleString()} cells`;
    const m=latest.meta.metrics,e=latest.meta.extras;
    $('metrics').textContent=`Frame ${latest.frameId} · Reduction ${m.cell_reduction_pct.toFixed(2)}% · Memory ${m.memory_saved_pct.toFixed(2)}% (est.) · RMSE ${m.obstacle_rmse_m.toFixed(4)} m · Ghosts ${m.ghost_cells_detected} · E2E ${latency.toFixed(1)} ms · Server ${e.server_fps?.toFixed(1)??'—'} FPS · Pack+send ${e.encode_ms?.toFixed(2)??'—'} ms (prior sample)`;
    if(e.runtime_config)$('speed').value=String(e.runtime_config.speed);
  }catch(error){showError(error);}
}
async function poll(){
  if(!polling)return;
  try{const response=await fetch('/api/frame',{headers:{Accept:'application/octet-stream'},cache:'no-store'});if(!response.ok)throw Error('REST '+response.status);receive(await response.arrayBuffer());$('connection').textContent='REST polling · Reconnecting';}catch(error){$('connection').textContent='Reconnecting';}
  pollTimer=setTimeout(poll,100);
}
function fallback(){if(!polling){polling=true;poll();}}
function connect(){
  $('connection').textContent='Connecting';
  try{socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/stream`);}catch(error){$('connection').textContent='Reconnecting';fallback();setTimeout(connect,retry);retry=Math.min(10000,retry*2);return;}socket.binaryType='arraybuffer';
  socket.onopen=()=>{retry=500;polling=false;clearTimeout(pollTimer);$('connection').textContent='WebSocket';};
  socket.onmessage=event=>{
    if(event.data instanceof ArrayBuffer){receive(event.data);return;}
    try{const message=JSON.parse(event.data);if(message.type==='error')showError(message.message);else if(message.type==='control_ack')playing=message.playing;}catch(error){showError(error);}
  };
  socket.onerror=()=>socket.close();
  socket.onclose=()=>{$('connection').textContent='Reconnecting';fallback();setTimeout(connect,retry);retry=Math.min(10000,retry*2);};
}
async function command(message){
  if(socket?.readyState===1){socket.send(JSON.stringify(message));return;}
  try{const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(message)});const result=await response.json();if(!response.ok)throw Error(result.detail);playing=result.playing;if(['step','reset'].includes(message.action)){const frame=await fetch('/api/frame',{headers:{Accept:'application/octet-stream'}});receive(await frame.arrayBuffer());}}catch(error){showError(error);}
}
for(const action of ['play','pause','step','reset'])$(action).onclick=()=>{if(action==='reset'){transform.reset();wipe=.5;$('wipe').style.left='50%';}command({action});};
$('home').onclick=()=>transform.reset();$('speed').onchange=event=>command({action:'configure',settings:{speed:Number(event.target.value)}});
canvas.onpointerdown=event=>{drag={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);};
canvas.onpointermove=event=>{if(drag){transform.pan(event.clientX-drag.x,event.clientY-drag.y);drag={x:event.clientX,y:event.clientY};}};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
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
  ctx.fillStyle='#050811';ctx.fillRect(0,0,width,height);
  if(latest){
    const extent=transform.extentRect();
    ctx.save();ctx.beginPath();ctx.rect(0,0,divider,height);ctx.clip();ctx.imageSmoothingEnabled=false;ctx.drawImage(raster,extent.x,extent.y,extent.width,extent.height);
    if(transform.pixelsPerCell>=6){ctx.strokeStyle='#49303a';ctx.lineWidth=.5;ctx.beginPath();for(let i=0;i<=GRID.dim;i++){const a=transform.gridToScreen(i,0),b=transform.gridToScreen(i,GRID.dim),c=transform.gridToScreen(0,i),d=transform.gridToScreen(GRID.dim,i);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.moveTo(c.x,c.y);ctx.lineTo(d.x,d.y);}ctx.stroke();}ctx.restore();
    ctx.save();ctx.beginPath();ctx.rect(divider,0,width-divider,height);ctx.clip();
    for(let i=0;i<latest.n;i++){const r=transform.cellRect(latest.x[i],latest.y[i],latest.level[i]);ctx.fillStyle=palette[latest.level[i]];ctx.fillRect(r.x,r.y,r.width,r.height);if(r.width>=4){ctx.strokeStyle='#668198';ctx.lineWidth=.4;ctx.strokeRect(r.x,r.y,r.width,r.height);}}ctx.restore();
    // World overlays intentionally sit outside both wipe clips.
    for(const track of latest.meta.dynamic_tracks){const point=transform.worldToScreen(track.x,track.y);ctx.strokeStyle='#9de1bc';ctx.lineWidth=1;ctx.strokeRect(point.x-4,point.y-4,8,8);ctx.fillStyle='#e5e9ef';ctx.font='11px ui-monospace,Consolas,monospace';ctx.fillText(`#${track.track_id}`,point.x+7,point.y);}
    const ego=transform.worldToScreen(0,0);ctx.fillStyle='#c9e4ee';ctx.beginPath();ctx.arc(ego.x,ego.y,4,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#b6c5d5';ctx.beginPath();ctx.moveTo(divider,0);ctx.lineTo(divider,height);ctx.stroke();drawCount++;
  }
  if(now-lastStats>=1000){const seconds=(now-lastStats)/1000;const fps=drawCount/seconds;receivedHz=received/seconds;$('fps').textContent=`Render ${fps.toFixed(1)} FPS · Received ${receivedHz.toFixed(1)} Hz`;Object.assign(window.viewerDiagnostics,{renderFps:fps,receivedFps:receivedHz});drawCount=0;received=0;lastStats=now;}
  requestAnimationFrame(draw);
}
connect();setTimeout(()=>{if(socket?.readyState!==1)fallback();},1500);requestAnimationFrame(draw);

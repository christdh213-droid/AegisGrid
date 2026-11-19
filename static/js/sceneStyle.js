/** Presentation only: all geometry and elevations come from the received frame. */
let dotTemplate=null;
export function paintBaseline(frame, canvas) {
  if(canvas.width!==1024)canvas.width=canvas.height=1024;
  const c=canvas.getContext('2d');
  if(!dotTemplate){dotTemplate=new Uint8ClampedArray(1024*1024*4);for(let y=0;y<256;y++)for(let x=0;x<256;x++){const i=((255-y)*4*1024+x*4+1025)*4;dotTemplate.set([116,145,163,55],i);}}
  const image=new ImageData(new Uint8ClampedArray(dotTemplate),1024,1024);
  for(let y=0;y<256;y++)for(let x=0;x<256;x++){
    const q=frame.raster[y*256+x];if(!q)continue;const z=-2+(q-1)*8/254;if(z<.12)continue;
    const t=Math.max(0,Math.min(1,(z-.16)/2)),color=[Math.round(155+90*t),Math.round(70+87*t),Math.round(63-30*t),z<.35?90:180];
    for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)image.data.set(color,(((255-y)*4+dy+1)*1024+x*4+dx+1)*4);
  }
  c.putImageData(image,0,0);
}
export function drawAdaptive(c,frame,t,mode) {
  const colors=['#173747','#123346','#102b41','#0f2544','#091428'];
  const risk=['#245f56','#977234','#a54848','#182638'];
  const elevated=new Path2D();
  for(let i=0;i<frame.n;i++){
    const r=t.cellRect(frame.x[i],frame.y[i],frame.level[i]),z=frame.zMax[i]/100;
    const occupied=!!(frame.flags[i]&8),height=Math.max(0,Math.min(1,(z-.35)/1.8));
    c.fillStyle=mode==='risk'?risk[(frame.flags[i]>>1)&3]:colors[frame.level[i]];
    if(mode==='elevation')c.fillStyle=occupied?`rgb(${Math.round(12+185*height)},${Math.round(25+84*height)},${Math.round(44-14*height)})`:'#070b14';
    c.globalAlpha=mode==='risk'?.85:.26;c.fillRect(r.x,r.y,r.width,r.height);c.globalAlpha=1;
    if(occupied && z>=.35 && frame.zMax[i]-frame.zMin[i]>12 && mode!=='risk'){
      c.fillStyle=`rgba(245,${Math.round(148-80*height)},${Math.round(25+25*height)},${.18+.5*height})`;
      c.fillRect(r.x+.3,r.y+.3,r.width-.6,r.height-.6);
      elevated.rect(r.x+.3,r.y+.3,r.width-.6,r.height-.6);
    }
    if(r.width>=4){c.strokeStyle='rgba(56,189,248,.28)';c.lineWidth=.65;c.strokeRect(r.x,r.y,r.width,r.height);
      c.fillStyle='rgba(180,228,255,.055)';c.fillRect(r.x,r.y,r.width,.8);}
  }
  if(mode!=='risk'){
    c.save();c.strokeStyle='#f59e0b70';c.lineWidth=.8;c.stroke(elevated);c.restore();
  }
}
export function drawRibbon(c,route,t,now,reduced){
  const points=route.planned_path??[],blocked=route.path_status==='BLOCKED';c.save();
  if(points.length){
    const path=new Path2D();points.forEach(([x,y],i)=>{const p=t.worldToScreen(x,y);i?path.lineTo(p.x,p.y):path.moveTo(p.x,p.y);});
    const first=t.worldToScreen(...points[0]),last=t.worldToScreen(...points.at(-1));
    const g=c.createLinearGradient(first.x,first.y,last.x,last.y);g.addColorStop(0,'#38bdf8');g.addColorStop(.55,'#46e4eb');g.addColorStop(1,'#6ee7b7');
    c.lineCap='round';c.lineJoin='round';
    if(blocked){c.setLineDash([8,9]);c.strokeStyle='#b8b7a880';c.lineWidth=3;c.stroke(path);}
    else {for(const [width,alpha,blur] of [[18,.12,22],[10,.24,12],[4,.95,7],[1.2,.9,0]]){c.lineWidth=width;c.globalAlpha=alpha;c.strokeStyle=g;c.shadowColor='#22d3ee';c.shadowBlur=blur;c.stroke(path);}}
  }
  if(!reduced && !blocked && points.length){c.globalAlpha=.65;c.shadowBlur=0;c.strokeStyle='#d5fff2';c.lineWidth=1.5;c.setLineDash([3,18]);c.lineDashOffset=reduced?0:-now/65;c.beginPath();points.forEach(([x,y],i)=>{const p=t.worldToScreen(x,y);i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y);});c.stroke();}
  c.globalAlpha=1;c.setLineDash([]);
  if(route.goal_xy){const p=t.worldToScreen(...route.goal_xy),pulse=reduced?0:(Math.sin(now/420)+1)*2;c.shadowColor='#34d399';c.shadowBlur=16;c.strokeStyle=blocked?'#fbbf24':'#6ee7b7';c.lineWidth=1.5;
    for(const radius of [7,13+pulse]){c.beginPath();c.arc(p.x,p.y,radius,0,Math.PI*2);c.stroke();}c.fillStyle='#cbffed';c.beginPath();c.arc(p.x,p.y,2.5,0,Math.PI*2);c.fill();}
  c.restore();
}
function rounded(c,x,y,w,h,r){c.beginPath();c.roundRect(x,y,w,h,r);}
export function drawActor(c,track,t){
  const p=t.worldToScreen(track.x,track.y),ped=track.kind==='PED',color=ped?'#67e8f9':'#fbbf24',scale=t.pixelsPerCell/.25;
  const vx=Number.isFinite(track.vx)?track.vx:0,vy=Number.isFinite(track.vy)?track.vy:0;
  const heading=Number.isFinite(track.heading)?track.heading:Math.atan2(vy,vx);
  c.save();c.lineCap='round';c.lineJoin='round';
  if(track.wake?.length>1){for(let i=1;i<track.wake.length;i++){const a=t.worldToScreen(...track.wake[i-1]),b=t.worldToScreen(...track.wake[i]);c.strokeStyle=color;c.globalAlpha=.3*i/track.wake.length;c.lineWidth=3;c.beginPath();c.moveTo(a.x,a.y);c.lineTo(b.x,b.y);c.stroke();}}c.globalAlpha=1;
  c.beginPath();c.moveTo(p.x,p.y);
  if(Array.isArray(track.predicted_path) && track.predicted_path.length>1){
    for(const xy of track.predicted_path){if(xy.length===2 && xy.every(Number.isFinite)){const q=t.worldToScreen(...xy);c.lineTo(q.x,q.y);}}
  }else{
    // Display-only constant-velocity 1.5 s projection; tangent uses measured heading.
    // Without a separate heading this is straight, avoiding an invented turn.
    const speed=track.speed,dx=Math.hypot(vx,vy)>0?vx:Math.cos(heading)*speed,dy=Math.hypot(vx,vy)>0?vy:Math.sin(heading)*speed;
    const end=t.worldToScreen(track.x+dx*1.5,track.y+dy*1.5),control=t.worldToScreen(track.x+Math.cos(heading)*speed*.75,track.y+Math.sin(heading)*speed*.75);
    c.quadraticCurveTo(control.x,control.y,end.x,end.y);
  }
  c.strokeStyle=color;c.shadowColor=color;c.shadowBlur=10;c.globalAlpha=.6;c.lineWidth=2;c.stroke();c.globalAlpha=1;
  c.translate(p.x,p.y);c.rotate(-heading+Math.PI/2);
  if(ped){
    c.shadowBlur=7;c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.ellipse(0,1,4,6,0,0,Math.PI*2);c.stroke();
    c.shadowBlur=0;c.fillStyle='#d6f5fa';c.beginPath();c.arc(0,-3,2.5,0,Math.PI*2);c.fill();c.strokeStyle='#abd8e3';c.beginPath();c.moveTo(-3,0);c.lineTo(3,0);c.moveTo(0,1);c.lineTo(-2,6);c.moveTo(0,1);c.lineTo(2,6);c.stroke();
  }else{
    const w=Math.max(11,Math.min(34,(track.width??1.5)*scale)),h=Math.max(w*1.65,Math.min(55,(track.length??3)*scale));
    c.strokeStyle='#fbbf2455';c.lineWidth=.8;rounded(c,-w/2-3,-h/2-3,w+6,h+6,5);c.stroke();c.shadowBlur=0;
    c.fillStyle='#02070e';for(const x of [-w/2-2,w/2-1])for(const y of [-h*.34,h*.2]){rounded(c,x,y,3,h*.18,1);c.fill();}
    const shell=c.createLinearGradient(-w/2,0,w/2,0);shell.addColorStop(0,'#2a4051');shell.addColorStop(.4,'#7894a4');shell.addColorStop(.55,'#526f82');shell.addColorStop(1,'#1b3144');c.fillStyle=shell;c.strokeStyle='#b8d5e0';
    rounded(c,-w/2,-h/2,w,h,Math.min(5,w/3));c.fill();c.stroke();
    c.fillStyle='#081a2a';rounded(c,-w*.36,-h*.28,w*.72,h*.48,3);c.fill();c.strokeStyle='#96bac780';c.stroke();
    c.fillStyle='#bbf1ff';c.fillRect(-w*.36,-h/2+2,w*.22,2);c.fillRect(w*.14,-h/2+2,w*.22,2);
    c.fillStyle='#ef785b';c.fillRect(-w*.36,h/2-3,w*.2,1.5);c.fillRect(w*.16,h/2-3,w*.2,1.5);
  }c.restore();
}
export function drawRadar(c,t,now=0,reduced=false){
  const p=t.worldToScreen(0,0);c.save();c.lineWidth=.75;c.font='10px system-ui';
  for(const radius of [10,20,30]){const r=radius*t.pixelsPerCell/.25;c.strokeStyle='#82cbe030';c.setLineDash([3,6]);c.beginPath();c.arc(p.x,p.y,r,0,Math.PI*2);c.stroke();c.fillStyle='#8bafc0';c.fillText(`${radius} m`,p.x+7,p.y-r+12);}
  c.setLineDash([]);const r=30*t.pixelsPerCell/.25;c.strokeStyle='#a5daec40';c.beginPath();for(let deg=0;deg<360;deg+=5){const a=deg*Math.PI/180,len=deg%30===0?6:2;c.moveTo(p.x+Math.cos(a)*r,p.y+Math.sin(a)*r);c.lineTo(p.x+Math.cos(a)*(r-len),p.y+Math.sin(a)*(r-len));}c.stroke();
  if(!reduced){const angle=now/9000;c.fillStyle='#6ed8de08';c.beginPath();c.moveTo(p.x,p.y);c.arc(p.x,p.y,r,angle-.25,angle);c.closePath();c.fill();c.strokeStyle='#8deddf33';c.beginPath();c.moveTo(p.x,p.y);c.lineTo(p.x+Math.cos(angle)*r,p.y+Math.sin(angle)*r);c.stroke();}
  c.restore();
}
export function drawEgo(c,t,now=0,reduced=false){
  const p=t.worldToScreen(0,0);c.save();c.translate(p.x,p.y);
  c.fillStyle='#b9f9ff0a';c.beginPath();c.moveTo(-5,-15);c.lineTo(-28,-62);c.lineTo(28,-62);c.lineTo(5,-15);c.closePath();c.fill();
  c.shadowColor='#67e8f9';c.shadowBlur=18;c.fillStyle='#70d8e12b';c.beginPath();c.ellipse(0,0,17,25,0,0,Math.PI*2);c.fill();c.shadowBlur=0;
  for(const x of [-12,7])for(const y of [-14,3]){c.fillStyle='#09121f';rounded(c,x,y,5,11,2);c.fill();c.strokeStyle='#658397';c.stroke();}
  const body=c.createLinearGradient(-8,0,8,0);body.addColorStop(0,'#38586b');body.addColorStop(.45,'#d2edf1');body.addColorStop(1,'#436278');c.fillStyle=body;rounded(c,-8,-17,16,34,5);c.fill();
  c.fillStyle='#102b3d';rounded(c,-6,-10,12,13,3);c.fill();c.strokeStyle='#8edbe8';c.stroke();
  c.fillStyle='#8bdee7';c.beginPath();c.arc(0,-2,3,0,Math.PI*2);c.fill();c.strokeStyle='#e0fffa';c.beginPath();c.arc(0,-2,5,reduced?0:now/500,(reduced?0:now/500)+2);c.stroke();c.strokeStyle='#d6fcff';c.beginPath();c.moveTo(0,-5);c.lineTo(0,-22);c.stroke();
  c.fillStyle='#d7fffb';c.fillRect(-6,-15,4,2);c.fillRect(2,-15,4,2);c.fillStyle='#163748';c.fillRect(-5,8,10,5);c.restore();
}
export function drawBeam(c,x,height){c.save();const g=c.createLinearGradient(x-9,0,x+9,0);g.addColorStop(0,'#67e8f900');g.addColorStop(.5,'#67e8f933');g.addColorStop(1,'#67e8f900');c.fillStyle=g;c.fillRect(x-9,0,18,height);c.shadowBlur=12;c.shadowColor='#67e8f9';c.strokeStyle='#b2eeffb3';c.lineWidth=1;c.beginPath();c.moveTo(x,0);c.lineTo(x,height);c.stroke();c.restore();}

/** Lightweight animation above a cached scene; no per-cell or full-map blur. */
export function drawActivity(c,route,t,now,reduced){
  if(reduced)return;const p=t.worldToScreen(0,0),r=30*t.pixelsPerCell/.25,angle=now/9000;
  c.save();c.fillStyle='#6ed8de08';c.beginPath();c.moveTo(p.x,p.y);c.arc(p.x,p.y,r,angle-.22,angle);c.closePath();c.fill();c.strokeStyle='#8deddf40';c.lineWidth=.7;c.beginPath();c.moveTo(p.x,p.y);c.lineTo(p.x+Math.cos(angle)*r,p.y+Math.sin(angle)*r);c.stroke();
  c.strokeStyle='#dcfff7';c.lineWidth=1.2;c.beginPath();c.arc(p.x,p.y-2,5,now/500,now/500+2);c.stroke();
  if(route.goal_xy){const q=t.worldToScreen(...route.goal_xy);c.strokeStyle='#6ee7b766';c.beginPath();c.arc(q.x,q.y,14+3*Math.sin(now/420),0,Math.PI*2);c.stroke();}
  if(route.planned_path?.length && route.path_status!=='BLOCKED'){c.lineWidth=1.5;c.strokeStyle='#e0fff3b0';c.setLineDash([3,19]);c.lineDashOffset=-now/65;c.beginPath();route.planned_path.forEach(([x,y],i)=>{const q=t.worldToScreen(x,y);i?c.lineTo(q.x,q.y):c.moveTo(q.x,q.y);});c.stroke();}
  c.restore();
}

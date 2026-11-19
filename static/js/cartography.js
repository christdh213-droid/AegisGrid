/** Offline cartography from the exact metric polygons used by the LiDAR source. */
function polygon(c,points,t,dx=0,dy=0){c.beginPath();points.forEach(([x,y],i)=>{const p=t.worldToScreen(x,y);i?c.lineTo(p.x+dx,p.y+dy):c.moveTo(p.x+dx,p.y+dy);});c.closePath();}
function line(c,points,t){c.beginPath();points.forEach(([x,y],i)=>{const p=t.worldToScreen(x,y);i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y);});c.stroke();}
export class ArchitectureLayer {
  constructor(){this.canvas=document.createElement('canvas');this.key='';this.layout=null;this.builds=0;}
  async load(){const response=await fetch('/api/scene_layout');if(!response.ok)throw Error('Scene layout unavailable');this.layout=await response.json();this.key='';}
  render(t,width,height,dpr){
    if(!this.layout)return null;
    const key=[width,height,dpr,t.width.toFixed(2),t.zoom,t.panX,t.panY].join(':');
    if(key===this.key)return this.canvas;
    this.key=key;this.builds++;this.canvas.width=width*dpr;this.canvas.height=height*dpr;
    const c=this.canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);
    const extent=t.extentRect();c.fillStyle='#090f1a';c.fillRect(extent.x,extent.y,extent.width,extent.height);
    const l=this.layout;
    for(const s of l.sidewalks){polygon(c,s.polygon,t);c.fillStyle='#1b283b';c.fill();c.strokeStyle='#7c98ad77';c.lineWidth=1;c.stroke();}
    for(const p of [...l.roads,...l.courtyards]){polygon(c,p,t);c.fillStyle='#0b111e';c.fill();}
    // Paint is planar at road Z=0, and never supplied to routing as an obstacle.
    c.fillStyle='#d6e5e8a8';for(const p of l.crosswalks){polygon(c,p,t);c.fill();}
    c.lineWidth=.8;c.strokeStyle='#e6b660bb';c.setLineDash([]);
    for(const sign of [-1,1])for(const offset of [-.16,.16]){
      line(c,[[sign*9,offset],[sign*31,offset]],t);line(c,[[offset,sign*9],[offset,sign*31]],t);
    }
    c.strokeStyle='#e3edf05c';c.lineWidth=.8;c.setLineDash([t.pixelsPerCell*6,t.pixelsPerCell*8]);
    for(const sign of [-1,1]){
      for(const lane of [-2.5,2.5]){line(c,[[sign*10,lane],[sign*31,lane]],t);line(c,[[lane,sign*10],[lane,sign*31]],t);}
    }
    c.setLineDash([]);c.strokeStyle='#e9f3f0c0';c.lineWidth=2;
    for(const sign of [-1,1]){line(c,[[sign*8,-4.6],[sign*8,-.5]],t);line(c,[[.5,sign*8],[4.6,sign*8]],t);}
    c.strokeStyle='#a8c1ca7a';c.lineWidth=1.1;
    for(const [x,y,angle] of [[-13,-2.5,0],[13,2.5,Math.PI],[-2.5,13,-Math.PI/2],[2.5,-13,Math.PI/2]]){
      const arrow=[[ -1,0],[1,0],[.2,.6],[1,0],[.2,-.6]].map(([a,b])=>[x+a*Math.cos(angle)-b*Math.sin(angle),y+a*Math.sin(angle)+b*Math.cos(angle)]);line(c,arrow,t);
    }
    for(const s of l.structures){
      const roof=s.kind==='building',lift=roof?s.height*t.pixelsPerCell*.33:0;
      polygon(c,s.polygon,t,3,5);c.fillStyle='#00000060';c.fill();
      polygon(c,s.polygon,t);c.fillStyle=roof?'#101b2a':s.kind==='parked_car'?'#516576':'#3b4a59';c.fill();c.strokeStyle='#8ca6b280';c.lineWidth=.8;c.stroke();
      if(roof){
        const ps=s.polygon;for(let i=0;i<ps.length;i++){const a=t.worldToScreen(...ps[i]),b=t.worldToScreen(...ps[(i+1)%ps.length]);c.beginPath();c.moveTo(a.x,a.y);c.lineTo(b.x,b.y);c.lineTo(b.x+lift*.35,b.y-lift);c.lineTo(a.x+lift*.35,a.y-lift);c.closePath();c.fillStyle=i%2?'#19293b':'#23354a';c.fill();}
        polygon(c,ps,t,lift*.35,-lift);const g=c.createLinearGradient(0,0,width,height);g.addColorStop(0,'#35485e');g.addColorStop(1,'#1c2e43');c.fillStyle=g;c.fill();c.strokeStyle='#8ca9bf88';c.lineWidth=1;c.stroke();
        c.save();c.clip();c.strokeStyle='#b7d9ee12';c.lineWidth=.6;const step=Math.max(12,t.pixelsPerCell*8);
        c.beginPath();for(let x=extent.x;x<extent.x+extent.width;x+=step){c.moveTo(x,extent.y);c.lineTo(x,extent.y+extent.height);}c.stroke();
        const cx=ps.reduce((n,p)=>n+p[0],0)/ps.length,cy=ps.reduce((n,p)=>n+p[1],0)/ps.length,p=t.worldToScreen(cx,cy),unit=t.pixelsPerCell*4;
        c.fillStyle='#102032';c.fillRect(p.x+lift*.35-unit,p.y-lift-unit/2,unit*2,unit);c.strokeStyle='#8aa9b366';c.strokeRect(p.x+lift*.35-unit,p.y-lift-unit/2,unit*2,unit);c.restore();
      }
      if(s.kind==='parked_car'){const xs=s.polygon.map(p=>p[0]),ys=s.polygon.map(p=>p[1]);const p=t.worldToScreen((Math.min(...xs)+Math.max(...xs))/2,(Math.min(...ys)+Math.max(...ys))/2);c.fillStyle='#0d1d2c';c.fillRect(p.x-t.pixelsPerCell*2,p.y-t.pixelsPerCell*4,t.pixelsPerCell*4,t.pixelsPerCell*8);}
    }
    return this.canvas;
  }
}

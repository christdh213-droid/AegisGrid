/** Single coordinate authority. Grid row 0 is y=-32; screen Y points down. */
export const GRID = Object.freeze({dim:256, resolution:0.25, half:32});
export class Transform {
  constructor(width=1,height=1) { this.width=width; this.height=height; this.zoom=1; this.panX=0; this.panY=0; }
  resize(width,height) { this.width=width; this.height=height; }
  get pixelsPerCell() { return Math.min(this.width,this.height)/GRID.dim*this.zoom; }
  worldToGrid(x,y) { return {col:Math.floor((x+GRID.half)/GRID.resolution),row:Math.floor((y+GRID.half)/GRID.resolution)}; }
  gridToWorld(col,row) { return {x:col*GRID.resolution-GRID.half,y:row*GRID.resolution-GRID.half}; }
  gridToScreen(col,row) { const s=this.pixelsPerCell; return {x:this.width/2+this.panX+(col-GRID.dim/2)*s,y:this.height/2+this.panY-(row-GRID.dim/2)*s}; }
  screenToGrid(x,y) { const s=this.pixelsPerCell; return {col:(x-this.width/2-this.panX)/s+GRID.dim/2,row:-(y-this.height/2-this.panY)/s+GRID.dim/2}; }
  worldToScreen(x,y) { return this.gridToScreen((x+GRID.half)/GRID.resolution,(y+GRID.half)/GRID.resolution); }
  screenToWorld(x,y) { const g=this.screenToGrid(x,y); return this.gridToWorld(g.col,g.row); }
  cellRect(col,row,level=0) { const size=2**level; const top=this.gridToScreen(col,row+size); return {x:top.x,y:top.y,width:size*this.pixelsPerCell,height:size*this.pixelsPerCell}; }
  extentRect() { return this.cellRect(0,0,8); }
  rasterIndex(col,row) { return ((GRID.dim-1-row)*GRID.dim+col)*4; }
  pan(dx,dy) { this.panX+=dx; this.panY+=dy; }
  zoomAt(factor,x,y) { const before=this.screenToGrid(x,y); this.zoom=Math.max(.5,Math.min(16,this.zoom*factor)); const after=this.gridToScreen(before.col,before.row); this.pan(x-after.x,y-after.y); }
  reset() { this.zoom=1; this.panX=0; this.panY=0; }
}

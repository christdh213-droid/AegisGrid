export function decodeFrame(buffer) {
  const view=new DataView(buffer);
  if(buffer.byteLength<40) throw Error('Truncated stream header');
  if(view.getUint32(0,true)!==0x4435324c || view.getUint16(4,true)!==1 || view.getUint16(6,true)!==40) throw Error('Unsupported stream version');
  const frameId=view.getUint32(8,true),serverTime=view.getFloat64(12,true);
  const nr=view.getUint32(20,true),na=view.getUint32(24,true),nj=view.getUint32(28,true),ne=view.getUint32(32,true),n=view.getUint32(36,true);
  if(nr!==65536 || na!==n*10 || 40+nr+na+nj+ne!==buffer.byteLength) throw Error('Invalid section lengths');
  let offset=40;
  const raster=new Uint8Array(buffer,offset,nr); offset+=nr;
  const x=new Uint16Array(buffer,offset,n);offset+=2*n;
  const y=new Uint16Array(buffer,offset,n);offset+=2*n;
  const level=new Uint8Array(buffer,offset,n);offset+=n;
  const flags=new Uint8Array(buffer,offset,n);offset+=n;
  const zMin=new Int16Array(buffer,offset,n);offset+=2*n;
  const zMax=new Int16Array(buffer,offset,n);offset+=2*n;
  const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,offset,nj)));
  return {frameId,serverTime,raster,n,x,y,level,flags,zMin,zMax,meta};
}

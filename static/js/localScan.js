import {GRID} from './transform.js';

export async function readLocalScan(file) {
  if (file.size > 8 * 1024 * 1024) throw Error('Choose a scan of 8 MiB or less.');
  const cells = new Float32Array(GRID.dim ** 2).fill(-Infinity);
  let count = 0;
  function add(x, y, z) {
    if (![x, y, z].every(Number.isFinite)) throw Error('The scan contains invalid coordinates.');
    const col = Math.floor((x + GRID.half) / GRID.resolution);
    const row = Math.floor((y + GRID.half) / GRID.resolution);
    if (col < 0 || row < 0 || col >= GRID.dim || row >= GRID.dim) return;
    const index = row * GRID.dim + col;
    cells[index] = Math.max(cells[index], z);
    count++;
  }
  if (/\.bin$/i.test(file.name)) {
    const buffer = await file.arrayBuffer();
    if (buffer.byteLength % 16) throw Error('BIN files must use little-endian float32 XYZI records.');
    const data = new DataView(buffer);
    for (let i = 0; i < data.byteLength; i += 16) {
      add(data.getFloat32(i, true), data.getFloat32(i + 4, true), data.getFloat32(i + 8, true));
    }
  } else if (/\.csv$/i.test(file.name)) {
    let first = true;
    for (const line of (await file.text()).split(/\r?\n/)) {
      if (!line.trim()) continue;
      const fields = line.split(',').map(field => field.trim());
      const header = first && ['x', 'y', 'z'].every((axis, i) => fields[i]?.toLowerCase() === axis);
      first = false;
      if (header) continue;
      if (fields.length < 3 || fields.slice(0, 3).some(field => field === '')) {
        throw Error('CSV files must contain x,y,z columns.');
      }
      add(Number(fields[0]), Number(fields[1]), Number(fields[2]));
    }
  } else {
    throw Error('Choose a .bin or .csv point cloud.');
  }
  if (!count) throw Error('The scan has no points inside the map bounds.');
  return {cells, count};
}

"""Session 1 wire, transform, frame-boundary and service regression checks."""
import asyncio
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.config import PipelineConfig, RuntimeConfig
from src.mapping_engine import FixedGridMap, AdaptiveQuadtreeMap
from src.pipeline import PerceptionPipeline
from src.stream_protocol import HEADER, Z_MIN, Z_MAX, Z_STEP, encode_frame, decode_frame
from src.stream_runtime import StreamState, updated_config
from prototype_app import app

ROOT = Path(__file__).resolve().parents[1]


def test_binary_round_trip():
    cfg = PipelineConfig()
    pipe = PerceptionPipeline(cfg)
    points = np.array([[-31.9, -31.9, -2], [0.1, 0.1, 1.456], [31.9, 31.9, 6.]])
    frame = pipe.process_frame(points)
    payload = encode_frame(frame, pipe.fixed_map, server_time_ms=123456.75)
    decoded = decode_frame(payload)
    assert encode_frame(frame, pipe.fixed_map, server_time_ms=123456.75, leaf_arrays=pipe.adaptive_map.leaf_arrays) == payload
    assert encode_frame(frame, pipe.fixed_map, server_time_ms=123456.75, leaf_arrays=pipe.adaptive_map.leaf_arrays, flags=pipe.adaptive_map.leaf_arrays['flags']) == payload
    header = HEADER.unpack_from(payload)
    assert header[:3] == (b'L25D', 1, 40)
    assert sum(header[5:9]) + 40 == len(payload)
    assert decoded['server_time_ms'] == 123456.75
    assert decoded['frame_id'] == frame.frame_id
    assert decoded['metadata']['dynamic_tracks'] == [t.to_dict() for t in frame.dynamic_tracks]
    grid = decoded['baseline'].reshape(256, 256)
    occupied = pipe.fixed_map.count > 0
    reconstructed = (grid[occupied].astype(float) - 1) * Z_STEP + Z_MIN
    assert np.max(np.abs(reconstructed - pipe.fixed_map.max_z[occupied])) <= Z_STEP
    assert np.all(grid[~occupied] == 0)
    for i, cell in enumerate(frame.adaptive_cells):
        cells = decoded['cells']
        assert cells['level'][i] == cell.level
        assert abs(cells['z_min'][i] / 100 - cell.min_z) <= .00501
        assert abs(cells['z_max'][i] / 100 - cell.max_z) <= .00501
        assert cells['flags'][i] >> 1 & 3 == 3  # UNKNOWN
    for invalid in (payload[:-1], b'wrong' + payload[5:], payload[:20]):
        with pytest.raises(ValueError):
            decode_frame(invalid)


def test_baseline_adaptive_alignment_and_js_decoder():
    """Exercise the production JS transform using an actual mapped known obstacle."""
    node = shutil.which('node')
    assert node, 'Node is required to run the production-transform alignment check'
    pipe = PerceptionPipeline(PipelineConfig())
    obstacle = (3.10, -7.10)
    pipe.process_frame(np.array([[*obstacle, 0.], [*obstacle, 1.5]]))
    col, row = int((obstacle[0] + 32) // .25), int((obstacle[1] + 32) // .25)
    cell = pipe.adaptive_map.leaves[pipe.adaptive_map.leaf_id_grid[row, col]]
    assert cell.level == 0
    transform_uri = (ROOT / 'static/js/transform.js').as_uri()
    protocol_uri = (ROOT / 'static/js/protocol.js').as_uri()
    # Binary also decoded in JS to catch offset/endianness drift between languages.
    import base64
    payload = encode_frame(pipe.process_frame(np.array([[*obstacle, 0.], [*obstacle, 1.5]])), pipe.fixed_map)
    code = f"""
      import assert from 'node:assert/strict';
      import {{Transform}} from {json.dumps(transform_uri)};
      import {{decodeFrame}} from {json.dumps(protocol_uri)};
      const t=new Transform(1200,800),point={json.dumps(obstacle)};
      const cell={json.dumps(cell.to_dict())};
      for(const [zoom,px,py] of [[1,0,0],[.5,45,-100],[3,-300,210],[16,13,17]]){{
        t.zoom=zoom;t.panX=px;t.panY=py;
        const grid=t.worldToGrid(...point),origin=t.worldToGrid(cell.x-cell.size/2,cell.y-cell.size/2);
        assert.deepEqual(t.cellRect(grid.col,grid.row),t.cellRect(origin.col,origin.row,cell.level));
        const screen=t.worldToScreen(...point),back=t.screenToWorld(screen.x,screen.y);
        assert.ok(Math.abs(back.x-point[0])<1e-10 && Math.abs(back.y-point[1])<1e-10);
        const before=t.screenToWorld(400,300);t.zoomAt(1.1,400,300);const after=t.screenToWorld(400,300);
        assert.ok(Math.abs(before.x-after.x)<1e-10 && Math.abs(before.y-after.y)<1e-10);
      }}
      assert.equal(t.rasterIndex(0,0),255*256*4);
      const b=Buffer.from('{base64.b64encode(payload).decode()}', 'base64');
      const f=decodeFrame(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
      assert.equal(f.frameId,1);assert.equal(f.n,{len(pipe.adaptive_map.leaves)});
    """
    subprocess.run([node, '--input-type=module'], input=code, check=True, capture_output=True, text=True)


def test_protected_mask_and_thresholds_are_per_build():
    cfg = PipelineConfig()
    fixed = FixedGridMap(cfg)
    fixed.update(np.array([[0.1, .1, .1], [.4, .1, .3]]))
    mapper = AdaptiveQuadtreeMap(cfg)
    mask = np.zeros((256, 256), dtype=bool)
    mapper.build(fixed, mask, merge_height_threshold=1., merge_gradient_threshold=1.)
    assert mapper.leaves[mapper.leaf_id_grid[128,128]].level > 0
    mask[128,128] = True
    mapper.build(fixed, np.zeros_like(mask), protected_mask=mask, merge_height_threshold=1., merge_gradient_threshold=1.)
    assert mapper.leaves[mapper.leaf_id_grid[128,128]].level == 0
    assert not mapper.leaves[mapper.leaf_id_grid[128,128]].is_dynamic
    assert cfg.merge_height_threshold == .12
    assert np.all(mapper.leaf_id_grid >= 0)


def test_controls_frozen_snapshot_defaults_and_validation():
    state = StreamState()
    state.execute({'action': 'step'})
    old = state.current_frame.extras['runtime_config']
    state.execute({'action': 'configure', 'settings': {'outlier_ratio': .15, 'sor_enabled': False}})
    assert old['outlier_ratio'] == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.runtime.speed = 2
    state.execute({'action': 'step'})
    assert state.current_frame.extras['runtime_config']['outlier_ratio'] == .15
    assert state.current_frame.extras['filtered_noise_points'] == 0
    assert state.current_frame.extras['sor_ms'] is None
    for updates in ({'outlier_ratio': .2}, {'speed': float('nan')}, {'sor_enabled': 'false'}, {'bogus': 1}, {'goal_xy': [1, float('inf')]}):
        with pytest.raises(ValueError):
            updated_config(state.runtime, updates)
    state.execute({'action': 'reset'})
    assert state.runtime == RuntimeConfig()
    assert state.current_frame.frame_id == 0


def test_latest_only_slot():
    state = StreamState()
    queue = asyncio.Queue(maxsize=1)
    state.subscribers.add(queue)
    for i in range(100):
        state.revision, state.latest_binary = i, bytes([i])
        state.publish()
    assert queue.qsize() == 1
    assert queue.get_nowait() == (99, bytes([99]))


def test_hook_order_and_custom_callback():
    log = []
    pipe = PerceptionPipeline(PipelineConfig(),
        source_hook=lambda p,m: (log.append('source') or p,m),
        pre_hook=lambda p,m: (log.append('pre') or p, {}),
        pre_map_hook=lambda f,c: log.append('pre_map') or {},
        post_hook=lambda c,cfg: (log.append('post_map') or c,{}),
        post_track_hook=lambda c,t,cfg: (log.append('post_track') or c,{}),
        post_frame_hook=lambda f,p: log.append('post_frame') or {},
        stream_callback=lambda f: log.append('callback'))
    pipe.process_frame(np.empty((0,3)))
    assert log == ['source','pre','pre_map','post_map','post_track','post_frame','callback']


def test_binary_rest_controls_legacy_and_websocket():
    with TestClient(app) as client:
        assert client.get('/legacy').status_code == 200
        assert client.get('/static/js/transform.js').status_code == 200
        assert client.post('/api/upload').status_code == 400  # Missing scan is rejected
        assert client.post('/api/control', json={'action':'configure','settings':{'outlier_ratio':1}}).status_code == 400
        assert client.post('/api/control', json=[]).status_code == 400
        response = client.post('/api/reset', headers={'Accept':'application/octet-stream'})
        assert decode_frame(response.content)['frame_id'] == 0
        with client.websocket_connect('/ws/stream') as ws:
            assert decode_frame(ws.receive_bytes())['frame_id'] == 0
            ws.send_json({'action':'configure','settings':{'sor_enabled':False}})
            assert ws.receive_json()['config']['sor_enabled'] is False
            ws.send_json({'action':'step'})
            binary = None
            ack = None
            for _ in range(2):
                message = ws.receive()
                if 'bytes' in message:
                    binary = decode_frame(message['bytes'])
                else:
                    ack = json.loads(message['text'])
            assert ack['action'] == 'step'
            assert binary['frame_id'] == 1
            assert binary['metadata']['extras']['runtime_config']['sor_enabled'] is False

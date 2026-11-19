import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
import prototype_app
from src.scan_upload import parse_scan, MAX_UPLOAD_BYTES, MAX_UPLOAD_POINTS
from src.stream_runtime import StreamState


def npy(points):
    output = io.BytesIO()
    np.save(output, points)
    return output.getvalue()


@pytest.mark.parametrize('extension', ['csv', 'bin', 'npy'])
def test_scan_formats_and_normalization(extension):
    points = np.array([[0, 0, -1.7, 1], [1, 1, -1.7, 1], [32, 32, -.7, 1], [40, 0, 100, 1]], dtype=np.float32)
    data = npy(points) if extension == 'npy' else points.tobytes() if extension == 'bin' else b'x,y,z,intensity\n0,0,-1.7,1\n1,1,-1.7,1\n32,32,-.7,1\n40,0,100,1'
    result, ground = parse_scan(data, 'scan.' + extension)
    assert result.shape == (3, 3)
    assert ground == pytest.approx(-1.7)
    assert result[:, 2] == pytest.approx([0, 0, 1])
    assert result[:, :2].max() < 32


@pytest.mark.parametrize('data,name', [(b'broken', 'x.bin'), (b'x,y,z\n0,nan,1', 'x.csv'), (npy(np.array([['bad', 'x', 'y']], dtype=object)), 'x.npy'), (npy(np.ones((1, 5))), 'x.npy'), (b'0,0,0', 'x.txt'), (b'40,40,0', 'x.csv')])
def test_invalid_scan(data, name):
    with pytest.raises(ValueError):
        parse_scan(data, name)


def test_scan_caps_before_array_allocation():
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr='<f8', fortran_order=False, shape=(MAX_UPLOAD_POINTS + 1, 3)))
    with pytest.raises(ValueError, match='points'):
        parse_scan(header.getvalue(), 'x.npy')
    with pytest.raises(ValueError, match='50 MiB'):
        parse_scan(b'0' * (MAX_UPLOAD_BYTES + 1), 'x.csv')


def test_static_scan_controls_and_return(monkeypatch):
    state = StreamState()
    state.disable_process_pool = True
    monkeypatch.setattr(prototype_app, 'state', state)
    points = np.array([[x, y, -1.7] for x in np.arange(-5, 5, .25) for y in np.arange(-5, 5, .25)])
    with TestClient(prototype_app.app) as client:
        response = client.post('/api/upload?filename=../../scan.npy', content=npy(points))
        assert response.status_code == 200
        assert state.current_frame.extras['source_mode'] == 'upload'
        assert state.current_frame.dynamic_tracks == []
        assert not state.pipeline.tracker.prev_dynamic_mask.any()
        assert state.current_frame.extras['safe_pct'] is not None
        client.post('/api/control', json={'action':'configure', 'settings':{'sor_enabled':False, 'simulate_hill_tilt':True, 'outlier_ratio':.15}})
        assert state.runtime.outlier_ratio == 0
        assert not state.runtime.simulate_hill_tilt
        assert state.current_frame.extras['runtime_config']['sor_enabled'] is False
        client.post('/api/control', json={'action':'set_goal', 'goal_xy':[2, 2]})
        assert state.current_frame.extras['path_status'] == 'OPTIMAL'
        client.post('/api/control', json={'action':'play'})
        assert not state.is_playing
        assert client.post('/api/upload?filename=x.csv', content=b'bad').status_code == 400
        client.post('/api/control', json={'action':'return_live'})
        assert state.current_frame.extras['source_mode'] == 'simulation'
        assert state.uploaded_points is None

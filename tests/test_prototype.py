"""
Unit & integration tests for prototype_app (FastAPI & WebSocket endpoints).
"""
import pytest
from fastapi.testclient import TestClient
from prototype_app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "DRDO SIH26053" in res.text
    assert "leftCanvas" in res.text
    assert "rightCanvas" in res.text


def test_api_frame(client):
    res = client.get("/api/frame")
    assert res.status_code == 200
    data = res.json()
    assert "frame_id" in data
    assert "adaptive_cells" in data
    assert "fixed_grid_b64" in data
    assert "metrics" in data
    assert len(data["fixed_grid_b64"]) > 50000


def test_api_step(client):
    res = client.post("/api/step")
    assert res.status_code == 200
    data = res.json()
    assert "frame_id" in data


def test_api_reset(client):
    res = client.post("/api/reset")
    assert res.status_code == 200
    data = res.json()
    assert data["frame_id"] == 0


def test_api_speed(client):
    res = client.post("/api/speed?fps=15")
    assert res.status_code == 200
    assert res.json()["fps"] == 15.0


def test_websocket_stream(client):
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert "frame_id" in data
        assert "adaptive_cells" in data
        assert "dynamic_tracks" in data

        ws.send_json({"action": "step"})
        data2 = ws.receive_json()
        assert "frame_id" in data2
        assert data2["frame_id"] >= data["frame_id"]


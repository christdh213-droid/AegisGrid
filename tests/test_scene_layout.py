"""Shared scene geometry and real point-cloud integration checks."""
import numpy as np
from fastapi.testclient import TestClient
from prototype_app import app
from src.config import PipelineConfig
from src.data_source import SyntheticLidarStream, scene_layout


def test_layout_endpoint_is_metric_geometry_authority():
    with TestClient(app) as client:
        response=client.get('/api/scene_layout')
    assert response.status_code == 200
    assert response.json() == scene_layout()
    layout=response.json()
    for structure in layout['structures']:
        xy=np.array(structure['polygon'])
        assert np.isfinite(xy).all() and (np.abs(xy)<=32).all()
        np.testing.assert_allclose(xy*100,np.round(xy*100),atol=1e-9)


def test_roads_goal_curbs_and_roofs_are_real_surfaces():
    source=SyntheticLidarStream(PipelineConfig())
    np.testing.assert_array_equal(source._surface_z(np.array([0.,-10.,-5.]),np.array([0.,8.,5.])),0.)
    assert source._surface_z(np.array([6.25]),np.array([16.]))[0] == .16
    assert source._surface_z(np.array([20.]),np.array([17.]))[0] == 4.2
    points,_=next(source)
    assert (points[:,2]>4.1).any()
    assert np.any(np.isclose(points[:,2],.16))


def test_reset_is_deterministic_and_four_actors_stay_on_corridors():
    source=SyntheticLidarStream(PipelineConfig())
    expected,metadata=next(source)
    source.reset();actual,replay=next(source)
    np.testing.assert_array_equal(actual,expected)
    assert metadata['ground_truth_actors'] == replay['ground_truth_actors']
    assert len(replay['ground_truth_actors']) == 4
    for actor in replay['ground_truth_actors']:
        if actor['type']=='VEHICLE': assert abs(actor['y'])==2.5 and actor['vy']==0
        else: assert abs(actor['x'])==6.25 and actor['vx']==0

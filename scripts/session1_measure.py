"""Session 1 recon only; not the final scenario/stress benchmark."""
import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_app import DashboardState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/session1_baseline_host.json')
    args = parser.parse_args()
    state = DashboardState()
    stages = {k: [] for k in ('generate', 'pre_process', 'fixed', 'track', 'adaptive', 'post_map', 'serialize', 'wall', 'payload_bytes')}
    def wrap(obj, name, key):
        original = getattr(obj, name)
        def measured(*a, **kw):
            start = time.perf_counter()
            result = original(*a, **kw)
            stages[key].append((time.perf_counter() - start) * 1000)
            return result
        setattr(obj, name, measured)
    wrap(state.stream, '_generate_frame', 'generate')
    for obj, name, key in ((state.pipeline, 'pre_hook', 'pre_process'),
                           (state.pipeline.fixed_map, 'update', 'fixed'),
                           (state.pipeline.tracker, 'process', 'track'),
                           (state.pipeline.adaptive_map, 'build', 'adaptive'),
                           (state.pipeline, 'post_hook', 'post_map')):
        wrap(obj, name, key)
    metrics = []
    for _ in range(300):
        start = time.perf_counter()
        data = state.step_frame()
        t = time.perf_counter()
        payload = json.dumps(data).encode()
        stages['serialize'].append((time.perf_counter() - t) * 1000)
        stages['wall'].append((time.perf_counter() - start) * 1000)
        stages['payload_bytes'].append(len(payload))
        metrics.append(data['metrics'])
    result = dict(frames=300, stages={k: dict(p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95))) for k, v in stages.items()},
                  throughput_fps=1000 / np.mean(stages['wall']),
                  golden=dict(max_rmse_m=max(m['obstacle_rmse_m'] for m in metrics),
                              min_cell_reduction_pct=min(m['cell_reduction_pct'] for m in metrics),
                              min_memory_saved_pct=min(m['memory_saved_pct'] for m in metrics),
                              ghost_cells_total=sum(m['ghost_cells_detected'] for m in metrics)))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

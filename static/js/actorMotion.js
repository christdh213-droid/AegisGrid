// Display-only dead reckoning. Server tracks remain authoritative and unchanged.
const MAX_EXTRAPOLATION_S = 0.25;
const CORRECTION_MS = 100;

export class ActorMotion {
  constructor() {
    this.tracks = new Map();
  }

  clear() {
    this.tracks.clear();
  }

  position(state, now) {
    const elapsed = Math.max(0, now - state.receivedAt);
    const seconds = Math.min(elapsed / 1000, MAX_EXTRAPOLATION_S);
    const t = Math.min(elapsed / CORRECTION_MS, 1);
    const correction = 1 - t * t * (3 - 2 * t);
    return {
      x: state.track.x + state.vx * seconds + state.offsetX * correction,
      y: state.track.y + state.vy * seconds + state.offsetY * correction,
    };
  }

  update(tracks, now) {
    const next = new Map();
    for (const track of tracks) {
      const key = track.track_id ?? track.label;
      const previous = this.tracks.get(key);
      const position = previous ? this.position(previous, now) : track;
      next.set(key, {
        track, receivedAt: now,
        vx: Number.isFinite(track.vx) ? track.vx : 0,
        vy: Number.isFinite(track.vy) ? track.vy : 0,
        offsetX: position.x - track.x,
        offsetY: position.y - track.y,
      });
    }
    this.tracks = next;
  }

  sample(now) {
    return Array.from(this.tracks.values(), state => ({
      ...state.track, ...this.position(state, now),
    }));
  }
}

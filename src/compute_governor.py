"""Host-scaled compute budgets with a quadtree resolution safety floor."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.ndimage import maximum_filter

from src.config import GOVERNOR_PROFILES
from src.stream_protocol import CELL_BYTES


@dataclass(frozen=True)
class GovernorSettings:
    height: float
    gradient: float
    replan_interval_s: float


class ComputeGovernor:
    ALPHA = .3
    STRESS_REFERENCE_LEAVES = 2400
    FIXED_BYTES_PER_CELL = 1

    def __init__(self, config, clock=perf_counter):
        self.config = config
        self.clock = clock
        self.reset()

    def reset(self):
        self.mode = None
        self.profile = None
        self.state = "NOMINAL"
        self.ema_ms = None
        self.over_frames = 0
        self.under_frames = 0
        self.fraction = 0.
        self.replan_interval_s = .1

    def select(self, mode):
        if mode != self.mode:
            self.reset()
            self.mode = mode
            self.profile = GOVERNOR_PROFILES[mode]
        height, gradient = self.config.merge_height_threshold, self.config.merge_gradient_threshold
        ceiling_h = self.profile.height_ceiling_m or height
        ceiling_g = self.profile.gradient_ceiling_m or gradient
        return GovernorSettings(
            height + self.fraction * max(0., ceiling_h - height),
            gradient + self.fraction * max(0., ceiling_g - gradient),
            self.replan_interval_s,
        )

    @staticmethod
    def protected_mask(fixed, previous_dynamic):
        mask = fixed.span_z > .35
        if previous_dynamic is not None:
            mask |= maximum_filter(previous_dynamic, size=5, mode="constant")
        return mask

    def map_options(self, fixed, previous_dynamic, settings):
        return {
            "merge_height_threshold": settings.height,
            "merge_gradient_threshold": settings.gradient,
            "protected_mask": self.protected_mask(fixed, previous_dynamic),
        }

    def inject_stress(self, leaf_count):
        """Perform work on this thread proportional to the number of leaves."""
        target = (self.profile.stress_ms / 1000) * leaf_count / self.STRESS_REFERENCE_LEAVES
        if target <= 0:
            return 0.
        start = self.clock()
        deadline = start + target
        while self.clock() < deadline:
            pass
        return (self.clock() - start) * 1000

    def observe(self, measured_ms, injected_ms, settings, leaf_count, fps):
        profile = self.profile
        latency = measured_ms * profile.cpu_scale + injected_ms
        self.ema_ms = latency if self.ema_ms is None else self.ema_ms + self.ALPHA * (latency - self.ema_ms)
        over = self.ema_ms > profile.budget_ms
        under = self.ema_ms <= .85 * profile.budget_ms
        self.over_frames = self.over_frames + 1 if over else 0
        self.under_frames = self.under_frames + 1 if under else 0
        if profile.height_ceiling_m is not None:
            if self.over_frames >= 2:
                if self.fraction < .99:
                    self.fraction += .4 * (1. - self.fraction)
                    if self.fraction >= .99:
                        self.fraction = 1.
                    self.state = "ADAPTING"
                else:
                    self.state = "BUDGET_LOCKED"
                    self.replan_interval_s = min(.4, self.replan_interval_s + .05)
            elif self.under_frames >= 4:
                self.state = "NOMINAL"
                if self.replan_interval_s > .1:
                    self.replan_interval_s = max(.1, self.replan_interval_s - .02)
                else:
                    self.fraction = max(0., self.fraction - .025)
            elif not over and self.state == "BUDGET_LOCKED":
                self.state = "ADAPTING"

        return {
            "governor_mode": self.mode,
            "governor_state": self.state,
            "target_budget_ms": profile.budget_ms,
            "governed_latency_ms": latency,
            "active_merge_threshold": settings.height,
            "active_gradient_threshold": settings.gradient,
            "radio_mb_s_fixed": (
                self.config.grid_dim ** 2 * self.FIXED_BYTES_PER_CELL * fps / 1e6 if fps is not None else None
            ),
            "radio_mb_s_adaptive": leaf_count * CELL_BYTES * fps / 1e6 if fps is not None else None,
        }

"""Sampling-based MPC expert: the direct-optimization alternative to the
hand-designed heuristic controllers in `flow_policy.expert`.

Every planning call, seeds a candidate action sequence from a heuristic
(see "Staged scoring" below for which one, and why), then refines it for a few rounds of the
cross-entropy method (CEM) -- sample perturbed candidate sequences, roll each
forward using the *real* physics engine (a separate "shadow" env kept in
exact sync with the live one), score each, keep the best few ("elites"),
refit the sampling distribution to them, repeat. Unlike a hand-tuned
heuristic, this directly optimizes toward the true objective using
ground-truth dynamics rather than an analytic proxy for what a good push
looks like.

Scoring is real coverage (`PushTEnv._get_coverage()`, the literal quantity
the 95%-coverage success criterion is defined on) plus small continuous
position/angle-error penalties, not coverage alone -- coverage is ~0 (flat,
no gradient) for any two poses that don't already substantially overlap, so
scoring on it alone gives CEM nothing to select on while the block is still
far from the goal; every candidate would score identically. The error
penalties are weighted small enough that coverage dominates once overlap
appears, so the search still refines toward that precisely rather than
stopping at "closer, but not overlapping."

**Staged scoring.** An early version scored candidates on their *final*
state only, seeded from the continuous-blend heuristic, over a short
horizon: it plateaued around the same coverage the heuristic alone gets
(~0.3-0.5), diagnosed as sometimes nailing position while leaving rotation
badly wrong (one seed ended 88 degrees off) or vice versa -- a short,
locally-perturbed-around-one-seed search doesn't reliably discover "fix
rotation first, then position" on its own. Fixed two ways: (1) the scoring
function adds a penalty on angle error at the *midpoint* of a longer
horizon, not just at the end, explicitly rewarding candidate sequences that
get rotation closer early rather than only checking where they land; (2) the
nominal seed sequence comes from the *hysteresis* expert (which behaviorally
already does rotate-then-translate) rather than continuous-blend, giving CEM
a starting point that embodies the staged strategy instead of having to
discover it purely through local perturbation.

The critical piece this depends on is exact state cloning into the shadow
env -- position, angle, *and* velocities for both bodies, not just the
5-dim observation, which drops velocity entirely. Verified empirically to
reproduce a live trajectory exactly under free motion and within ~0.05px
under active contact (small residual almost certainly from the physics
engine's contact-solver warm-starting, not a bug) -- see git history for the
validation script. One non-obvious gotcha this relies on getting right:
pymunk rotates a body around its `center_of_gravity` when `.angle` is set,
which is nonzero for this block, so setting `.angle` after `.position` moves
`.position` again as a side effect -- `_restore` below sets angle first.

**Known limitation, unresolved.** Even after the fixes above plus a
persisted nominal-heuristic instance (so its rotate/translate commitment
carries across planning calls, not reset every cycle) and a minimum-
improvement gate (a candidate must beat the nominal by a real margin, not
just any positive epsilon, to be accepted), this still doesn't reliably beat
the plain heuristics: mean coverage across a 12-episode/variant batch was
~0.11-0.17, worse than continuous-blend alone (~0.3-0.4) and worse than an
earlier, less-fixed version of this same file (~0.2-0.25). Traced one
concrete failure: CEM found a candidate sequence that scored a genuine
0.05+ improvement over the nominal *within its 16-step lookahead*, accepted
it (correctly, per the gate), and that locally-better choice left the agent
stuck circling without ever re-approaching the block for the rest of a
300-step episode. That's finite-horizon greedy search finding a locally
attractive trajectory that's globally worse -- a real limitation of this
design, not a bug with an obvious patch. Untried next steps: a shaped value
function that accounts for what happens *after* the horizon rather than
scoring only the lookahead window, or a much larger search budget than
tested here. Left in the repo as working-but-not-reliable infrastructure --
not wired into data collection, not used by default anywhere.
"""

import dataclasses

import numpy as np

from flow_policy.envs import TaskVariant, make_variant_env
from flow_policy.expert import ScriptedExpertConfig, ScriptedPushTExpert, _wrap_angle

# Coverage is ~0 for any two poses that don't already substantially overlap --
# a flat, uninformative fitness landscape for CEM candidates that are still far
# from the goal (every candidate scores 0, so there's nothing to select on).
# Blending in small continuous position/angle-error penalties gives CEM a
# gradient to climb even before any overlap exists; their weights are small
# enough that once overlap *does* appear, coverage dominates and the search
# refines toward it precisely rather than getting stuck at "close enough".
_POS_ERROR_WEIGHT = 0.001  # per px of origin-to-goal distance, at the final step
_ANGLE_ERROR_WEIGHT = 0.05  # per radian of angle error, at the final step
_MID_ANGLE_ERROR_WEIGHT = 0.1  # per radian of angle error, at the horizon's midpoint (see module docstring)


def _snapshot(env) -> dict:
    u = env.unwrapped
    return {
        "agent_pos": tuple(u.agent.position),
        "agent_vel": tuple(u.agent.velocity),
        "block_angle": u.block.angle,
        "block_pos": tuple(u.block.position),
        "block_vel": tuple(u.block.velocity),
        "block_angular_vel": u.block.angular_velocity,
    }


def _restore(env, snap: dict) -> None:
    u = env.unwrapped
    u.agent.position = snap["agent_pos"]
    u.agent.velocity = snap["agent_vel"]
    u.block.angle = snap["block_angle"]  # angle before position -- see module docstring
    u.block.position = snap["block_pos"]
    u.block.velocity = snap["block_vel"]
    u.block.angular_velocity = snap["block_angular_vel"]


@dataclasses.dataclass
class MPCExpertConfig:
    horizon: int = 16  # control steps planned ahead per CEM search
    num_candidates: int = 32  # sampled sequences per CEM iteration
    num_elites: int = 8
    num_cem_iters: int = 3
    action_noise_std: float = 60.0  # px; initial per-step sampling std around the nominal sequence
    noise_decay: float = 0.5  # std multiplier after each CEM iteration
    # A candidate must beat the nominal by more than this to be accepted, not just any
    # positive epsilon. Without this, CEM would chase marginal or noisy score gains
    # (e.g. from a glancing, unproductive contact) over the nominal's own coherent
    # rotate-then-translate strategy -- locally better within the horizon, but
    # empirically capable of leaving the agent stuck somewhere unproductive for the
    # rest of the episode. Coverage/pos/angle-error are on roughly a [0,1]-ish combined
    # scale, so 0.05 is a real, not noise-level, improvement.
    min_improvement: float = 0.05
    # Execute the *whole* planned horizon before replanning, not a prefix of it: with
    # persisted heuristic-phase state (see MPCExpert.__init__), simulating a full
    # horizon for the nominal seed but only executing part of it for real would leave
    # the heuristic's phase reflecting a hypothetical trajectory further along than
    # physical reality -- executing the full plan keeps the two in lockstep.
    replan_every: int = 16


class MPCExpert:
    """Direct-optimization expert via CEM search over a synced shadow env.

    Unlike `flow_policy.expert.ScriptedPushTExpert`, this needs the *live env
    object* each step (to read exact physics state, not just the 5-dim
    observation, which drops velocity) -- so its `act` takes `env`, not `obs`,
    and it cannot be used as a drop-in replacement wherever the other experts
    are used. Owns a persistent shadow env for simulation, separate from
    whatever env is actually driving the episode.
    """

    def __init__(self, variant: TaskVariant, config: MPCExpertConfig | None = None, seed: int | None = None):
        self.variant = variant
        self.goal_pose = variant.goal_pose
        self.config = config or MPCExpertConfig()
        self._shadow_env = make_variant_env(variant)
        self._shadow_env.reset()
        self._cached_plan: list[np.ndarray] = []
        self._steps_since_replan = 0
        # CEM sampling was drawing from numpy's global random state -- the same
        # episode seed could (and did, empirically) produce wildly different CEM
        # search outcomes (0.0 vs 0.82 final coverage) depending on unrelated prior
        # calls elsewhere. Own RNG, seeded explicitly, makes a given (env seed,
        # mpc seed) pair fully reproducible.
        self._rng = np.random.default_rng(seed)
        # Persisted, not recreated each planning call: a *fresh* hysteresis instance
        # every cycle starts back at phase="translate" and only re-enters "rotate" if
        # the angle error is already above rotation_enter_thresh -- discarding whatever
        # commitment it had built up. That defeats hysteresis's whole point (finish
        # rotating before switching back), and empirically caused the nominal seed to
        # flip-flop between cycles instead of committing to a rotate-then-translate
        # maneuver, leaving the agent circling without ever re-approaching the block
        # for an entire episode. Reusing one instance carries its phase across
        # planning calls, much closer to how it behaves run continuously.
        self._nominal_heuristic = ScriptedPushTExpert(self.goal_pose, ScriptedExpertConfig(rotation_mode="hysteresis"))

    def reset(self) -> None:
        self._cached_plan = []
        self._steps_since_replan = 0
        self._nominal_heuristic = ScriptedPushTExpert(self.goal_pose, ScriptedExpertConfig(rotation_mode="hysteresis"))

    def close(self) -> None:
        self._shadow_env.close()

    def _rollout_score(self, snap: dict, actions) -> float:
        cfg = self.config
        _restore(self._shadow_env, snap)
        u = self._shadow_env.unwrapped
        mid_step = len(actions) // 2
        mid_angle_error = 0.0
        obs = None
        for i, action in enumerate(actions):
            obs, _, terminated, truncated, _ = self._shadow_env.step(np.clip(action, 0.0, 512.0).astype(np.float32))
            if i == mid_step - 1:
                mid_angle_error = abs(_wrap_angle(self.goal_pose[2] - obs[4]))
            if terminated or truncated:
                break
        coverage = float(u._get_coverage())
        pos_error = float(np.linalg.norm(obs[2:4] - self.goal_pose[:2]))
        angle_error = abs(_wrap_angle(self.goal_pose[2] - obs[4]))
        return (
            coverage
            - _POS_ERROR_WEIGHT * pos_error
            - _ANGLE_ERROR_WEIGHT * angle_error
            - _MID_ANGLE_ERROR_WEIGHT * mid_angle_error
        )

    def _nominal_sequence(self, snap: dict) -> np.ndarray:
        _restore(self._shadow_env, snap)
        # Reuses the persisted hysteresis instance (see __init__) rather than a fresh
        # one, so its rotate/translate commitment carries across planning calls.
        heuristic = self._nominal_heuristic
        obs = self._shadow_env.unwrapped.get_obs()
        actions = []
        for _ in range(self.config.horizon):
            action = heuristic.act(obs)
            actions.append(action.astype(np.float64))
            obs, _, terminated, truncated, _ = self._shadow_env.step(action)
            if terminated or truncated:
                break
        while len(actions) < self.config.horizon:
            actions.append(actions[-1].copy())
        return np.stack(actions)  # (horizon, 2)

    def _plan(self, env) -> list[np.ndarray]:
        cfg = self.config
        snap = _snapshot(env)
        nominal = self._nominal_sequence(snap)
        mean = nominal
        std = np.full_like(mean, cfg.action_noise_std)

        best_seq = mean
        best_score = self._rollout_score(snap, mean)

        for _ in range(cfg.num_cem_iters):
            noise = self._rng.standard_normal((cfg.num_candidates, *mean.shape)) * std[None, :, :]
            candidates = np.clip(mean[None, :, :] + noise, 0.0, 512.0)
            scores = np.array([self._rollout_score(snap, c) for c in candidates])

            # If no candidate this round differs meaningfully from any other (e.g. none
            # made contact within the horizon, so coverage/pos/angle-error -- all purely
            # functions of the block's pose -- are identical for every candidate), the
            # "elite" selection below is really just an arbitrary tie-break, not a
            # meaningful choice. Averaging those arbitrary picks can drag `mean` into a
            # degenerate region (empirically: a candidate sequence collapsing to the
            # same repeated action for the whole horizon, parking the agent somewhere it
            # never re-approaches the block). Skip the update when there's no real signal
            # to act on instead of trusting the tie-break.
            if scores.max() - scores.min() < 1e-9:
                continue

            top = np.argsort(scores)[-cfg.num_elites :]
            if scores[top[-1]] > best_score + cfg.min_improvement:
                best_score = float(scores[top[-1]])
                best_seq = candidates[top[-1]]
            mean = candidates[top].mean(axis=0)
            std = std * cfg.noise_decay

        # Belt-and-suspenders: if whatever CEM settled on is a near-constant action
        # across the whole horizon (the specific degenerate shape observed in practice)
        # and it doesn't even clearly beat the nominal, don't trust it -- fall back to
        # the nominal sequence, which is at least known to keep circling/re-engaging.
        best_seq_arr = np.asarray(best_seq)
        is_degenerate = best_seq_arr.std(axis=0).max() < 1.0
        if is_degenerate and best_score <= self._rollout_score(snap, nominal) + 1e-6:
            best_seq = nominal

        return list(best_seq)

    def act(self, env) -> np.ndarray:
        if not self._cached_plan or self._steps_since_replan >= self.config.replan_every:
            self._cached_plan = self._plan(env)
            self._steps_since_replan = 0
        action = self._cached_plan[self._steps_since_replan]
        self._steps_since_replan += 1
        return np.clip(action, 0.0, 512.0).astype(np.float32)

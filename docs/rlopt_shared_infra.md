# RLOPT Shared Infrastructure

The shared RL-optimization infrastructure lets the `t1_so3`, `t2_se3`, and
`p0_grasp` environment agents enable dense rewards and derived state features
**through YAML configuration only**.  No environment agent edits this code.

All components are **off by default**: with an empty `features` and empty
`reward_components` section, the encoded state vector, reward, config, demo
cache identity, and resume behavior are identical to the frozen RLE0 recipe
semantics (the on-disk `config.json` additionally records the two empty
sections; behavior and metrics are unchanged).

## Modules

| File | Role |
|---|---|
| `rlinf/envs/dynamic_benchmark/geometry.py` | Causal pose/quaternion helpers (SO(3) geodesic, relative pose, finite-difference EE twist). |
| `rlinf/envs/dynamic_benchmark/feature_registry.py` | Fixed-length derived state features with mask compatibility. |
| `rlinf/envs/dynamic_benchmark/reward_registry.py` | Config-driven dense reward components with independent recompute. |
| `examples/embodiment/train_dynamic_benchmark_expert.py` | Trainer wiring: YAML-only nested sections, infra identity, resume tolerance. |
| `examples/embodiment/probe_rlopt_infra.py` | CPU-only probe: enabled features/rewards, shapes, masks, no-leakage, recompute error. |
| `examples/embodiment/verify_rlopt_infra_parity.py` | Standalone parity gate (default-off byte parity, no future leakage, recompute < 1e-7, identity stability). |

## 1. State feature registry (`features`)

Config schema: a mapping from feature name to either `true` or a mapping of
parameters.  Default is `{}` (all features off).

| Feature | Shape | Parameters | Prerequisites (privileged keys) | Notes |
|---|---|---|---|---|
| `relative_pose` | 7 | — | `object_pose_wxyz`, `eef_pose_xyzw` | Object pose in the EE frame: translation + wxyz quaternion. |
| `geodesic_error` | 1 | — | `object_pose_wxyz`, `goal_pose_wxyz` | SO(3) geodesic distance (rad) between object and goal orientation. |
| `object_vel` | 6 | — | `object_twist_world` | Object twist in world frame. |
| `ee_vel` | 6 | — | `eef_pose_xyzw` | Causal finite-difference EE twist; first step after reset is masked. |
| `relative_vel` | 6 | — | `object_twist_world`, `eef_pose_xyzw` | `ee_vel - object_vel` (world frame). |
| `belt_speed` | 1 | — | `belt_surface_velocity_geom` | Norm of the belt surface velocity vector. |
| `time_to_goal` | 1 | — | `object_pose_wxyz`, `eef_pose_xyzw`, `object_twist_world` | `distance / max(closing_speed, 0.01)` clamped to `[0, 60]` s. |
| `stage` | 1 | — | — | Current `active_stage_progress` in `[0, 1]`; masked at reset. |
| `action_history` | `7*k` | `k` (int, `1..8`, default `3`) | — | Last `k` executed actions, oldest first, zero-padded; masked until one action exists. |
| `goal_error` | 1 | — | `object_pose_wxyz`, `goal_pose_wxyz` | Translation distance (m) between object and goal position. |

Rules:

- Every feature has a fixed length; missing prerequisites are encoded as zeros
  with a `0` mask entry (same masking contract as optional base fields).
- Features are appended *after* the frozen base state vector, so the base
  schema and its byte-identical default path are untouched.
- No future information is consumed: only the current observation, previously
  executed actions, and the previous EE pose (finite difference, fixed
  `dt = 0.05 s` from the 20 Hz policy-rate contract).
- Unknown feature names, invalid `k`, or unknown parameters fail closed at
  config parse time.

Example:

```yaml
features:
  relative_pose: true
  geodesic_error: true
  object_vel: true
  action_history:
    k: 3
```

## 2. Dense reward registry (`reward_components`)

Config schema: a mapping from component name to either a weight number or a
mapping with a required `weight` key plus optional component parameters.
`weight` must be finite and non-negative; `0.0` disables the component.  Raw
component values are already signed (penalties <= 0, bonuses >= 0).

| Component | Raw value (times weight) | Parameters (defaults) | Inputs |
|---|---|---|---|
| `r_ori_geodesic` | `-min(geodesic_error / scale_rad, 1)` | `scale_rad` (1.0) | `geodesic_error_rad` |
| `r_rel_pose` | `-min(trans/scale_pos + rot/scale_rot, 1)` | `scale_pos_m` (0.1), `scale_rot_rad` (1.0) | `relative_translation_error_m`, `relative_rotation_error_rad` |
| `r_effort` | `-action_l2` | — | `action_l2` |
| `r_completion_shaping` | `-scale * max(0, 1 - completion)` when `completion >= near_threshold`, else 0 | `near_threshold` (0.9), `scale` (1.0) | `completion` |
| `r_vel_align` | `-relative_velocity_norm / speed_scale` | `speed_scale_m_s` (1.0) | `relative_velocity_norm_m_s` |
| `r_timing` | `-min(abs(ttc - target_ttc)/ttc_scale, 1)` when `distance <= dist_threshold`, else 0 | `dist_threshold_m` (0.05), `target_ttc_s` (0.1), `ttc_scale_s` (0.1) | `distance_m`, `time_to_goal_s` |
| `r_stage` | `stage_progress - previous_stage_progress` | — | `stage_progress`, `previous_stage_progress` (registry state) |

Rules:

- Every component is a pure function of the per-step inputs assembled by the
  environment from the current observation, executed action, and previous
  stage progress.  The full input set is recorded in
  `info["reward_inputs"]["registry"]` for independent recompute.
- Independent recompute re-runs the component functions on the recorded
  inputs; the gate requires max absolute error `< 1e-7` (float64).
- No future information: only the current state/action and the previous stage
  progress (persisted for exact resume) are used.
- Unknown components, unknown parameters, missing `weight`, or invalid values
  fail closed.

Example:

```yaml
reward_components:
  r_ori_geodesic:
    weight: 2.0
    scale_rad: 0.5
  r_effort:
    weight: 0.005
```

## 3. Identity, checkpoint, and resume contract

- `config.json`, `summary.json`, policy files, trainer checkpoints, and demo
  cache identities record `infra_identity` whenever features or reward
  components are enabled: the feature registry dict + SHA-256, and the reward
  registry dict + SHA-256.
- With default-empty sections, `infra_identity` is `null`/omitted so legacy
  demo caches and frozen checkpoints remain loadable (controlled legacy
  upgrade, same policy as the canonical `reward_safety_penalty` cache fix).
- Resuming a checkpoint requires an exact config/state-schema match; missing
  legacy `features`/`reward_components` keys are treated as empty.
- Different feature sets or reward weights produce different
  `state_schema`/`infra_identity`/demo identities, so every arm gets an
  independent run identity.  Same configuration reruns produce identical
  behavior and metrics.
- `DynamicBenchmarkEnv` checkpoints persist registry states, action histories,
  previous EE poses, EE velocities, and stage progresses for bit-exact resume
  (checkpoint schema v0.3; v0.2 checkpoints load only when infra is default).

## 4. Training-acceleration switches (reused from RLE-U1)

These switches already exist in the base line (`267e1140`, derived from
`5e4ed6e9` process workers and `ff11d168`/`3cbf3779` borrowed evaluation).
They are exposed as flat YAML keys / CLI flags:

| Switch | YAML key / CLI flag | Default | Effect |
|---|---|---|---|
| Training process workers | `env_worker_processes` / `--env-worker-processes` | 0 (serial) | Persistent subprocess shards for reset/step. |
| Eval process workers | `eval_worker_processes` / `--eval-worker-processes` | 0 (serial) | Persistent subprocess shards for validation. |
| Planner in eval processes | `eval_planner_in_processes` / `--eval-planner-in-processes` | false | Runs the residual planner inside the eval subprocess. |
| Persistent eval pool | `persistent_eval_workers` / `--persistent-eval-workers` | false | Keeps the eval pool alive between validation calls. |
| Borrowed eval | `borrow_training_env_for_eval` / `--borrow-training-env-for-eval` | false | Runs validation on the training pool with manifest switch/restore (~9x faster 20K validation in RLE-U1). |
| Sampler/learner overlap | `sampler_learner_overlap` / `--sampler-learner-overlap` | false | Overlaps process-backed sampling with learner updates. |
| 100K budget | `total_env_steps` / `--total-env-steps` | 200000 | Budget template used by RLE-U1 (100000). |
| Instrumentation | `timing_sample_interval` / `--timing-sample-interval` | 50 | Phase timing for update/sample rates; telemetry also covers GPU/CPU via the RLE-U1 supervisor. |

Validation constraints (fail closed): process workers require matching
`worker_threads=1`; borrowed eval requires matching train/eval width and
worker topology, and is exclusive with persistent eval and process planner.
Every switch defaults to the frozen serial behavior.

Templates:

- `examples/embodiment/config/rlopt_speed_100k_process8.yaml`
- `examples/embodiment/config/rlopt_speed_borrowed_eval.yaml`

## 5. Example recipes

- `examples/embodiment/config/rlopt_t1_so3.yaml` — geodesic/reference-pose
  shaping + relative/error features (initial recommendation for RLOPT-SO3).
- `examples/embodiment/config/rlopt_t2_se3.yaml` — effort + completion shaping
  (initial recommendation for RLOPT-SE3).
- `examples/embodiment/config/rlopt_p0_grasp.yaml` — velocity alignment +
  timing + stage shaping with belt/velocity features (initial recommendation
  for RLOPT-P0G).

Numeric weights in the examples are **recommendations only**; the environment
agents own the final values and must not change this module to tune them.

## 6. Verification

```bash
# CPU-only probe for a recipe
python examples/embodiment/probe_rlopt_infra.py \
  --config examples/embodiment/config/rlopt_t1_so3.yaml \
  --output /tmp/rlopt_probe.json

# CPU-only parity gate (default-off byte parity, no leakage, recompute < 1e-7)
python examples/embodiment/verify_rlopt_infra_parity.py \
  --output /tmp/rlopt_gate.json
```

Unit tests: `tests/unit_tests/test_dynamic_benchmark_reward_registry.py`,
`tests/unit_tests/test_dynamic_benchmark_feature_registry.py`,
`tests/unit_tests/test_dynamic_benchmark_infra_parity.py`.

## 7. Common pitfalls

- Do not put `features`/`reward_components` on the CLI; they are YAML-only
  nested sections.  Putting them in a recipe YAML is required.
- Changing a feature set or reward weight **changes the run identity**; do not
  mix arms under one run root or resume across identities.
- `ee_vel`/`relative_vel`/`time_to_goal` are masked for the first step after a
  reset (no previous pose).  This is expected and does not leak information.
- The base reward terms (`reward_step_penalty`, `reward_action_l2_scale`,
  `reward_safety_penalty`) are not registry components; keep tuning them via
  the existing YAML keys.
- Consumption must always use hash-pinned **best** policies; enabling features
  does not change that contract.

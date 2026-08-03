RL with Dynamic Benchmark
=========================

.. figure:: /_static/svg/dynamic-benchmark.svg
   :align: center
   :width: 90%

   Dynamic Benchmark task families and the privileged-state expert workflow.

Dynamic Benchmark is a research-only MuJoCo suite for dynamic Franka manipulation.
RLinf trains compact expert policies from current privileged state and online planner
demonstrations, then exports audited trajectories for downstream world-action modeling.

Overview
--------

Train resumable BC, SAC, RLPD, planner-residual RLPD, or PPO experts across 14
grasping, placement, interaction, and replanning tasks.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP actor · 10-head REDQ critic

   .. grid-item-card:: Algorithms
      :text-align: center

      BC · SAC · RLPD-SAC · residual RLPD · PPO

   .. grid-item-card:: Tasks
      :text-align: center

      T1–T5 · P0-Grasp · 14 total

   .. grid-item-card:: Hardware
      :text-align: center

      1 GPU per independent run

| **You'll do:** install the private benchmark source → run the T2 RLPD recipe → inspect success and safety gates.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a local SE3-WAM checkout with the ``benchmark`` extra.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 28 58

   * - Family
     - Config IDs
     - Focus
   * - T1
     - ``t1_belt``, ``t1_xyz``, ``t1_so3``, ``t1_occ``
     - Conveyor grasping under motion, pose variation, rotation, and occlusion.
   * - T2
     - ``t2_trans``, ``t2_se3``
     - Grasp and place into moving translation-only or full SE(3) goals.
   * - T3
     - ``t3_phase``, ``t3_full``
     - Timed interaction under phase-visible and full dynamic regimes.
   * - T4
     - ``t4_sphere``, ``t4_slider``, ``t4_can``, ``t4_sphere_tabletop``
     - Release, capture, and contact-sensitive object interactions.
   * - T5
     - ``t5_replan``
     - Replan after an observed dynamic event without future-state leakage.
   * - P0
     - ``p0_grasp``
     - Dynamic grasp with bilateral contact, verified hold, clearance, and stable lift.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Specification
   * - Observation
     - 173-D current privileged state: 138 values plus a 35-D availability mask. Camera observations are disabled for state-policy training.
   * - Action
     - Normalized 7-D E7 command: 3 translation deltas, 3 rotation deltas, and gripper control.
   * - Reward
     - Recomputable potential progress, success, safety, timeout, step, and action-effort terms under schema ``rlinf-dynamic-benchmark-reward-v0.2``.
   * - Prompt
     - None. The task ID and current state define the expert-policy input.

Installation
------------

.. include:: _setup_common.rst

Set ``SE3_WAM_PATH`` to the authorized local checkout, then install the environment-only
bundle. The installer fails closed when the source is missing.

.. code-block:: bash

   export SE3_WAM_PATH=/path/to/SE3-WAM
   bash requirements/install.sh embodied --env dynamic_benchmark
   source .venv/bin/activate

The ``embodied-dynamic_benchmark`` Docker target installs only public runtime
dependencies. Mount and install the research-only SE3-WAM checkout as described in
``docker/README.md``.

Run It
------

Launch the frozen T2 RLPD recipe with explicit source identity and an output directory:

.. code-block:: bash

   export SE3_WAM_PATH=/path/to/SE3-WAM
   RLINF_COMMIT=$(git rev-parse HEAD)
   BENCHMARK_COMMIT=$(git -C "$SE3_WAM_PATH" rev-parse HEAD)
   python examples/embodiment/train_dynamic_benchmark_expert.py \
      --config examples/embodiment/config/dynamic_benchmark_t2_rlpd.yaml \
      --rlinf-commit "$RLINF_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --output outputs/dynamic_benchmark/t2_rlpd_seed1

**What this command does:**

1. Collects success-only planner demonstrations online; it does not consume an external dataset.
2. Runs BC warm-start followed by RLPD-SAC with an equal demo/online replay mixture.
3. Evaluates on a frozen validation manifest and checkpoints model, replay, environment, optimizer, normalizer, and RNG state.

The optional ``--actor-bc-weight`` adds a demonstration behavior-cloning loss to
each online actor update. Its default is ``0`` so the reference RLPD recipe remains
an unregularized baseline; record nonzero values as separate experiment arms.

Set ``--algorithm residual_rlpd`` to execute
``clamp(planner_action + residual_scale * policy_residual, -1, 1)``. Replay and the
critic stay in residual-action space, planner demonstrations map to the exact zero
residual, and checkpoints include the stateful planner instances. ``--residual-scale``
defaults to ``0.25``; treat each scale and actor-BC weight as a separate arm.

The matched on-policy control is a separate resumable trainer:

.. code-block:: bash

   python examples/embodiment/train_dynamic_benchmark_ppo.py \
      --config examples/embodiment/config/dynamic_benchmark_t1_ppo.yaml \
      --rlinf-commit "$RLINF_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --output outputs/dynamic_benchmark/t1_ppo_seed1

It uses a squashed-Gaussian 7-D actor, GAE, clipped PPO updates, a value head, frozen
validation manifests, and update-boundary checkpoint/resume. Time-limit truncations
bootstrap the value target but stop advantage recursion across the reset boundary.

Fresh BC/RLPD runs write ``demo_replay.pt`` after planner collection. Pass that file
with ``--demo-replay-in`` to reuse demonstrations across matched algorithm or
regularization arms. Loading fails closed unless the source commits, task, state
schema, seed, manifest, environment count, and demonstration contract match exactly;
the replay sampling state, normalizer, and post-collection RNG state are restored.

.. warning::

   These experts use privileged simulator state. Treat their trajectories as teacher
   data, not as deployable vision-policy results. Keep ``--rlinf-commit`` and
   ``--benchmark-commit`` at full 40-character hashes; resume rejects identity drift.

Evaluate the frozen best checkpoint on a deterministic test-ID or test-OOD manifest
with a separately identified evaluator commit:

.. code-block:: bash

   POLICY=outputs/dynamic_benchmark/t2_rlpd_seed1/best_policy.pt
   POLICY_SHA=$(sha256sum "$POLICY" | cut -d' ' -f1)
   EVALUATOR_COMMIT=$(git rev-parse HEAD)
   python examples/embodiment/evaluate_dynamic_benchmark_expert.py \
      --policy "$POLICY" \
      --expected-policy-sha256 "$POLICY_SHA" \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --rlinf-commit "$RLINF_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --split test_id \
      --manifest-seed 20261250 \
      --episodes 20 \
      --output outputs/dynamic_benchmark/t2_rlpd_seed1/test_id

The evaluator reconstructs BC/SAC/RLPD, residual-RLPD, and PPO policies, verifies the
checkpoint and source identities, records the reset manifest and executed actions,
requires exact action replay for every episode, and reports deterministic success,
safety, completion, effort, and decision-latency metrics. Keep test manifests unread
until validation-based policy and hyperparameter selection is frozen. Use
``--split validation`` for evaluator engineering smoke tests before that freeze.

Evaluate the privileged planner on the same paired validation manifests before the
freeze, without loading a learned policy:

.. code-block:: bash

   python examples/embodiment/evaluate_dynamic_benchmark_planner.py \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --task t1_xyz \
      --split validation \
      --manifest-seed 20261150 \
      --episodes 8 \
      --output outputs/dynamic_benchmark/t1_xyz_planner/validation_seed1

The planner evaluator stores the same reset identity, executed actions, exact replay
evidence, task metrics, and decision-latency gate. Every action tape is replayed on an
independently constructed raw environment, and the replay reset restores task-hidden
runtime state such as the T5 event tape before exact observation, outcome, and final
state checks. Test-ID/OOD remain available only for the single post-freeze comparison.

Export best-known trajectories
------------------------------

After validation freezes the candidate pool, put exactly one planner at candidate
index zero and the hash-pinned policies after it in a
``rlinf-dynamic-benchmark-optimal-candidates-v0.1`` JSON manifest. Each policy row
records its path, SHA-256, stochastic flag, exploration seed offset, and optional
residual-scale override. Exporters use a frozen 8→16→32 escalation budget and select
the first stable winner under success, safety, completion, return, control steps, and
action effort:

.. code-block:: bash

   CANDIDATES=outputs/dynamic_benchmark/t2_candidates.json
   CANDIDATES_SHA=$(sha256sum "$CANDIDATES" | cut -d' ' -f1)
   python examples/embodiment/export_dynamic_benchmark_optimal_trajectories.py \
      --candidate-manifest "$CANDIDATES" \
      --expected-candidate-manifest-sha256 "$CANDIDATES_SHA" \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --rlinf-commit "$RLINF_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --split train --manifest-seed 20261050 \
      --accepted-episodes 100 --max-resets 200 \
      --output outputs/dynamic_benchmark/t2_optimal_v1

An interrupted export can repeat the same command with ``--resume``. It verifies the
immutable source and candidate identities, preserves any uncommitted tail in a
sibling recovery directory, truncates JSONL files to the last atomically committed
reset boundary, and reruns that reset. Every attempt keeps a lightweight
state/action/reward tape and exact replay evidence; each winner additionally keeps
RGB-D/HDF5 evidence.

Run the independent auditor before consuming the dataset. Pass the final
``dataset_card.json`` and ``checksums.sha256`` hashes printed by the exporter:

.. code-block:: bash

   python examples/embodiment/audit_dynamic_benchmark_optimal_trajectories.py \
      --dataset-root outputs/dynamic_benchmark/t2_optimal_v1 \
      --expected-dataset-card-sha256 "$DATASET_CARD_SHA" \
      --expected-checksums-sha256 "$CHECKSUMS_SHA" \
      --expected-candidate-manifest-sha256 "$CANDIDATES_SHA" \
      --auditor-commit "$EVALUATOR_COMMIT" \
      --output outputs/dynamic_benchmark/t2_optimal_v1.audit.json

The auditor independently recomputes checksums, tape shapes and hashes, scores,
escalation, winner selection, exact benchmark replay, and HDF5/lightweight action
parity. Only a passing audit writes ``training_eligible=true``. Here ``optimal`` means
best-known within the immutable candidate/reset/budget contract, not a proof of
global continuous-control optimality.

Visualization and Results
-------------------------

Read ``metrics.jsonl`` while training and ``summary.json`` after completion. Rank
policies lexicographically by success, safety, completion, return, duration, and
action effort. The run also writes ``best_policy.pt``, ``final_policy.pt``, and
``checkpoint_latest.pt``. When a scheduled validation already represents the final
environment step, the trainer reuses it and records ``validation_reused`` instead of
running an identical evaluation twice. No reference success-rate claim is published until the
multi-seed benchmark screen completes.

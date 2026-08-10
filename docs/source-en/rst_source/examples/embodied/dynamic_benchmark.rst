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

The optional ``--reward-safety-penalty`` controls the terminal safety-failure reward
used for training and defaults to ``-10``. It must be finite and non-positive, is
recorded in the run and demonstration-cache identities, and every non-default value
must be treated as a separate validation-only experiment arm. Evaluation keeps its
canonical reward contract so return values remain comparable across trained policies.

Set ``--algorithm residual_rlpd`` to execute
``clamp(planner_action + residual_scale * policy_residual, -1, 1)``. Replay and the
critic stay in residual-action space, planner demonstrations map to the exact zero
residual, and checkpoints include the stateful planner instances. ``--residual-scale``
defaults to ``0.25``; treat each scale and actor-BC weight as a separate arm.

Environment stepping can use persistent subprocess shards instead of the serial or
threaded adapter. Set ``--eval-worker-processes 2`` (or ``4``/``8``) to accelerate
the frozen-manifest validation loop, and ``--env-worker-processes`` to shard training
reset/step calls. The parent process still assigns manifest rows and restores replies
by environment index, so seed and episode order do not depend on worker completion
order. Each subprocess is serial internally; keep the corresponding
``--*-worker-threads`` value at ``1``. ``--process-start-method spawn`` is the
portable, CUDA-safe default. Process count is execution provenance and therefore a
checkpoint must be resumed with the same configuration.

Before a throughput bakeoff, run the process correctness gate against the real
benchmark checkout. It compares serial/process reset and step digests, verifies an
exact process-mode checkpoint/resume suffix, and intentionally crashes one worker to
check bounded cleanup of every shard:

.. code-block:: bash

   python examples/embodiment/verify_dynamic_benchmark_process_runtime.py \
      --task t4_sphere --num-envs 8 --worker-processes 4 \
      --output outputs/dynamic_benchmark/process_gate.json

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

RLD2-QA calibration and formal evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The production evaluation schema is selected by the frozen-threshold arguments.
Supplying both ``--quality-v2-thresholds`` and
``--expected-quality-v2-thresholds-sha256`` produces expert evaluation schema
``rlinf-dynamic-benchmark-expert-evaluation-v0.3`` or planner evaluation schema
``rlinf-dynamic-benchmark-planner-evaluation-v0.2``. Each record is bound to a
canonical ``rlinf-dynamic-benchmark-optimal-attempt-v0.3`` tape. The planner-only
``metric_calibration`` wave deliberately omits those arguments and remains on
``rlinf-dynamic-benchmark-planner-evaluation-v0.1``; it is a separate scientific
partition, not a production policy comparison.

Before freezing Qv3 thresholds, run the exact-14 × exact-20 planner wave from a
clean evaluator checkout. The scheduler must perform the resource conflict check
before handing one to eight GPU indices to the launcher:

.. code-block:: bash

   RUNTIME_ROOT=/path/to/runtime
   RUNS_ROOT=/path/to/runs
   export RLD2_QA_BENCHMARK_SOURCE_ROOT="$RUNTIME_ROOT/SE3-WAM"
   bash examples/embodiment/rld2_qa_planner_calibration.sh \
      rld2qa-cal 0,1,2,3 "$PWD" "$RUNTIME_ROOT" \
      "$RUNS_ROOT" /path/to/tmp

The launcher predeclares all 280 reset identities before any rollout, evaluates
20 rows for every task in the canonical 14-task order, and verifies the selected
safe successful planner trajectory in three distinct fresh environments. With the
default wave ID it writes the canonical receipt at
``$RUNS_ROOT/RLD2-QA/planner-calibration-metric-v03-s20261350/wave_receipt.json``.
The frozen ``se3-wam-trajectory-quality-v2-thresholds-v0.3`` contract must bind
that exact receipt and remain disjoint from validation, review, test-ID, and
test-OOD manifests.

After the threshold contract is frozen, evaluate a checkpoint on the paired
20-row validation manifest with a separately identified evaluator commit:

.. code-block:: bash

   POLICY=outputs/dynamic_benchmark/t2_rlpd_seed1/best_policy.pt
   POLICY_SHA=$(sha256sum "$POLICY" | cut -d' ' -f1)
   QUALITY_V2_THRESHOLDS=/path/to/quality_v2_thresholds.json
   QUALITY_V2_SHA=$(sha256sum "$QUALITY_V2_THRESHOLDS" | cut -d' ' -f1)
   EVALUATOR_COMMIT=$(git rev-parse HEAD)
   python examples/embodiment/evaluate_dynamic_benchmark_expert.py \
      --policy "$POLICY" \
      --expected-policy-sha256 "$POLICY_SHA" \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --rlinf-commit "$RLINF_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --quality-v2-thresholds "$QUALITY_V2_THRESHOLDS" \
      --expected-quality-v2-thresholds-sha256 "$QUALITY_V2_SHA" \
      --split validation \
      --manifest-seed 20261150 \
      --episodes 20 \
      --output outputs/dynamic_benchmark/t2_rlpd_seed1/validation_formal

The evaluator reconstructs BC/SAC/RLPD, residual-RLPD, and PPO policies, verifies the
checkpoint and source identities, records the reset manifest and executed actions,
uses separate once-reset raw environments for each rollout and replay, and requires
exact action replay for every episode. Formal output recursively seals its attempt
tapes, Qv3 summaries and gates, replay receipts, and source hashes. For T5-Replan the
tape also records the canonical issued/applied action history and the causal latency
from impact end to the first qualifying applied correction. Keep test manifests
unread until validation-based policy and hyperparameter selection is frozen.

Evaluate the privileged planner on the same paired manifest and threshold identity,
without loading a learned policy:

.. code-block:: bash

   python examples/embodiment/evaluate_dynamic_benchmark_planner.py \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --task t2_trans \
      --split validation \
      --manifest-seed 20261150 \
      --episodes 20 \
      --quality-v2-thresholds "$QUALITY_V2_THRESHOLDS" \
      --expected-quality-v2-thresholds-sha256 "$QUALITY_V2_SHA" \
      --output outputs/dynamic_benchmark/t2_trans_planner/validation_formal

The planner evaluator stores the same reset identity, executed actions, exact replay
evidence, task metrics, and decision-latency gate. For every manifest row, both the
planner rollout and action-tape replay use separately constructed raw environments
that are reset exactly once. The replay reset also restores task-hidden runtime state
such as the T5 event tape before exact observation, outcome, and final-state checks.
The threshold-bearing command above emits planner schema v0.2; planner v0.1 is
confined to the exact-14 calibration launcher.
Test-ID/OOD remain available only for the single post-freeze comparison.

Export best-known trajectories
------------------------------

First generate the paired Owner-review subset from an
``rlinf-dynamic-benchmark-optimal-candidates-v0.1`` review candidate manifest whose
``review_contract`` has ``full_generation=false``, whose planner is candidate zero,
and whose learned candidates carry passing v0.2 promotion receipts:

``build_dynamic_benchmark_rld2_promotion.py`` must receive the same calibration
sidecar for every v0.2 promotion through
``--quality-v2-calibration-wave-receipt`` and
``--expected-quality-v2-calibration-wave-receipt-sha256`` in addition to its frozen
threshold path/SHA and other required inputs. It reopens the sidecar and seals its
file/payload identity plus dataset-relative binding into both selection evidence and
the promotion receipt. The review exporter requires and revalidates the same pair.

.. code-block:: bash

   CALIBRATION_RECEIPT="$RUNS_ROOT/RLD2-QA/planner-calibration-metric-v03-s20261350/wave_receipt.json"
   CALIBRATION_RECEIPT_SHA=$(sha256sum "$CALIBRATION_RECEIPT" | cut -d' ' -f1)
   REVIEW_CANDIDATES=outputs/dynamic_benchmark/t2_trans_review_candidates.json
   REVIEW_CANDIDATES_SHA=$(sha256sum "$REVIEW_CANDIDATES" | cut -d' ' -f1)
   python examples/embodiment/export_dynamic_benchmark_rld2_review.py \
      --candidate-manifest "$REVIEW_CANDIDATES" \
      --expected-candidate-manifest-sha256 "$REVIEW_CANDIDATES_SHA" \
      --quality-v2-thresholds "$QUALITY_V2_THRESHOLDS" \
      --expected-quality-v2-thresholds-sha256 "$QUALITY_V2_SHA" \
      --quality-v2-calibration-wave-receipt "$CALIBRATION_RECEIPT" \
      --expected-quality-v2-calibration-wave-receipt-sha256 "$CALIBRATION_RECEIPT_SHA" \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --evaluator-benchmark-commit "$BENCHMARK_COMMIT" \
      --partition review --manifest-seed 20261250 --review-resets 20 \
      --output outputs/dynamic_benchmark/t2_trans_review

The review exporter writes schema
``rlinf-dynamic-benchmark-rld2-paired-review-v0.2``, evaluates the full candidate
pool deterministically, and selects six paired categories per task. Its receipts
always record ``full_generation=false``; review selection is not trajectory
eligibility.

.. warning::

   Do not start accepted-100 production generation or the full release build before
   the Owner explicitly approves the paired review for each task. There is no review
   CLI flag that grants or substitutes for that approval.

After Owner approval, production input is an exact-14
``rlinf-dynamic-benchmark-rld2-candidate-release-v0.2``. Each task manifest is
``rlinf-dynamic-benchmark-optimal-candidates-v0.2`` at the canonical
``<release-root>/<task>/candidate_manifest.json`` path; candidate zero is the planner,
and every candidate has hash-pinned provenance. Production selection must use the
full pool and ``planner-pareto``:

.. code-block:: bash

   CANDIDATE_RELEASE_ROOT=outputs/dynamic_benchmark/rld2_candidates
   CANDIDATES="$CANDIDATE_RELEASE_ROOT/t2_trans/candidate_manifest.json"
   CANDIDATE_RELEASE="$CANDIDATE_RELEASE_ROOT/release_manifest.json"
   CANDIDATES_SHA=$(sha256sum "$CANDIDATES" | cut -d' ' -f1)
   CANDIDATE_RELEASE_SHA=$(sha256sum "$CANDIDATE_RELEASE" | cut -d' ' -f1)
   python examples/embodiment/export_dynamic_benchmark_optimal_trajectories.py \
      --candidate-manifest "$CANDIDATES" \
      --expected-candidate-manifest-sha256 "$CANDIDATES_SHA" \
      --candidate-release-manifest "$CANDIDATE_RELEASE" \
      --expected-candidate-release-manifest-sha256 "$CANDIDATE_RELEASE_SHA" \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --evaluator-benchmark-commit "$BENCHMARK_COMMIT" \
      --quality-v2-thresholds "$QUALITY_V2_THRESHOLDS" \
      --expected-quality-v2-thresholds-sha256 "$QUALITY_V2_SHA" \
      --quality-v2-calibration-wave-receipt "$CALIBRATION_RECEIPT" \
      --expected-quality-v2-calibration-wave-receipt-sha256 "$CALIBRATION_RECEIPT_SHA" \
      --split train --manifest-seed 20261050 \
      --accepted-episodes 100 --max-resets 200 \
      --candidate-search-mode full-pool \
      --selection-mode planner-pareto \
      --output outputs/dynamic_benchmark/t2_trans_optimal_v2

The frozen threshold contract derives the per-task 10- or 11-dimension Qv3 comparison
instead of hard-coding one global phase inventory. A learned attempt may replace the
same-reset planner only when it passes success, safety, exact replay, the Qv3 absolute
gate, and the T5 causal gate when applicable; is non-worse on every frozen task/Qv3/
duration/control/effort/causal dimension; and is strictly better on at least one of
them. The planner wins an exact tie, and return is diagnostic only.

The exporter requires both the frozen threshold path/SHA and the authoritative
calibration receipt source path/SHA. It reopens the canonical exact-14 × exact-20
receipt, cross-checks its task and reset identities with the v0.3 threshold binding,
and copies it to the safe dataset-relative ``provenance/.../wave_receipt.json`` path
recorded by that contract.

An interrupted export can repeat the same command with ``--resume``. It verifies the
immutable source and candidate identities, preserves any uncommitted tail in a
sibling recovery directory, truncates JSONL files to the last atomically committed
reset boundary, and reruns that reset. Every attempt keeps a lightweight
state/action/reward tape and exact replay evidence; each winner additionally keeps
RGB-D/HDF5 evidence.

If the independently selected lightweight winner fails canonical render replay, the
reset is rejected instead of being published. New exports bind a structured
``render_parity_skip`` record to the selected attempt, while the sealed recovery log
retains the failure event. Sharded exports must be sealed with
``merge_optimal_export_shards.py`` so those events survive prefix truncation. The
auditor accepts a skipped reset only when it independently selects the same attempt,
finds no published winner, and validates the matching recovery evidence; skipped
resets never count toward the accepted quota.

Run the independent auditor before consuming the dataset. Hash the final
``dataset_card.json`` and root ``SHA256SUMS`` files and pass the frozen Qv3 threshold
identity as well:

.. code-block:: bash

   DATASET_ROOT=outputs/dynamic_benchmark/t2_trans_optimal_v2
   DATASET_CARD_SHA=$(sha256sum "$DATASET_ROOT/dataset_card.json" | cut -d' ' -f1)
   CHECKSUMS_SHA=$(sha256sum "$DATASET_ROOT/SHA256SUMS" | cut -d' ' -f1)
   python examples/embodiment/audit_dynamic_benchmark_optimal_trajectories.py \
      --dataset-root "$DATASET_ROOT" \
      --expected-dataset-card-sha256 "$DATASET_CARD_SHA" \
      --expected-checksums-sha256 "$CHECKSUMS_SHA" \
      --expected-candidate-manifest-sha256 "$CANDIDATES_SHA" \
      --expected-quality-v2-thresholds-sha256 "$QUALITY_V2_SHA" \
      --auditor-commit "$EVALUATOR_COMMIT" \
      --output outputs/dynamic_benchmark/t2_trans_optimal_v2.audit.json

The auditor independently recomputes checksums, tape shapes and hashes, scores,
full-pool Pareto selection, T5 issued/applied causal evidence, exact benchmark replay,
and HDF5/lightweight action parity. It also reopens the dataset-local calibration
receipt and verifies its canonical bytes, dataset-relative path, and threshold SHA
binding. Only a passing audit writes ``training_eligible=true``. Here ``optimal``
means best-known within the immutable candidate/reset/budget contract, not a proof
of global continuous-control optimality.

Visualization and Results
-------------------------

Read ``metrics.jsonl`` while training and ``summary.json`` after completion. Rank
policies lexicographically by success, safety, completion, return, duration, and
action effort. The run also writes ``best_policy.pt``, ``final_policy.pt``, and
``checkpoint_latest.pt``. When a scheduled validation already represents the final
environment step, the trainer reuses it and records ``validation_reused`` instead of
running an identical evaluation twice. No reference success-rate claim is published until the
multi-seed benchmark screen completes.

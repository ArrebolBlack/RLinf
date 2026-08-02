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

Train resumable BC, SAC, or RLPD experts across 14 grasping, placement, interaction,
and replanning tasks.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP actor · 10-head REDQ critic

   .. grid-item-card:: Algorithms
      :text-align: center

      BC · SAC · RLPD-SAC

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

.. warning::

   These experts use privileged simulator state. Treat their trajectories as teacher
   data, not as deployable vision-policy results. Keep ``--rlinf-commit`` and
   ``--benchmark-commit`` at full 40-character hashes; resume rejects identity drift.

Visualization and Results
-------------------------

Read ``metrics.jsonl`` while training and ``summary.json`` after completion. Rank
policies lexicographically by success, safety, completion, return, duration, and
action effort. The run also writes ``best_policy.pt``, ``final_policy.pt``, and
``checkpoint_latest.pt``. No reference success-rate claim is published until the
multi-seed benchmark screen completes.

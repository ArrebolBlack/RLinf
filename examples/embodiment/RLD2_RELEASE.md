# RLD2 exact-14 trajectory release record

RLD2 evaluates a frozen RL checkpoint pool against the task-specific planner on
the Dynamic Benchmark validation split. The release target is 100 accepted
best-known trajectories for each of the following tasks, with no partial-task
release:

`p0_grasp`, `t1_xyz`, `t1_belt`, `t1_so3`, `t1_occ`, `t2_trans`, `t2_se3`,
`t3_phase`, `t3_full`, `t4_sphere`, `t4_sphere_tabletop`, `t4_slider`,
`t4_can`, and `t5_replan`.

## Frozen scientific contract

- Candidate release manifest SHA-256:
  `80a19dfbc912b2cab3d594f3e105b24767541b3c46f3dbe7c1cfd29f0ff55a13`.
- Candidate release `SHA256SUMS` SHA-256:
  `438471c2eed7ca6059f699422240fb9ce1e103cfd5b28c38fe877613f511d90c`.
- Scientific evaluator RLinf commit:
  `d0c1e0ecb1a1e09aa6b44162b1fec26f2af5dbb9`.
- Evaluator SE3-WAM commit:
  `518c13bcd43880350f467da0f16ebfda9642b544`.
- Split and manifest seed: `validation`, `20260806`.
- Search and selection: full candidate pool for every reset, followed by the
  calibrated `planner-pareto` selection contract.
- Per-task target and reset ceiling: 100 accepted trajectories within 200
  frozen resets.
- Frozen test data is not used for model selection. The release remains
  training-ineligible until the per-task auditors and independent unified
  release auditor both pass.

The a08 launch gate remains bound to the evaluator commit above. The LF-stable
JSON writer merged later at `98f9b26ee7043729c5e73fa528b020b90e8494ed`
was proven byte-identical for all 151 runner JSON artifacts, so it did not
change or invalidate a08 evidence.

## Gate and attempt history

- `a08`: passed all 87 checkpoint-compatibility probes and all 14×3 planner
  calibration probes. Its package and evidence remain immutable.
- `b01`: failed closed before environment execution because production
  candidate identity included compatibility relations while planner
  calibration intentionally binds only the evaluator core identity. Commit
  `fd08bb124c3056ea46269c3c3471b2f291b556df` fixed that projection without
  changing the scientific utility or selection contract.
- `b02`: failed closed at reset zero. It exposed an inactive task-quality
  backend, vectorized `[None]` task-quality handling, and mixed compatible
  checkpoint state schemas. Commit
  `ed7de9a18571ce52133bcc30ef87feb5384a3201` fixed those runner issues without
  changing the scientific contract.
- `b03`: began valid single-process full-pool evaluation but used only one CPU
  core per lane. It was stopped through the official wrapper signal path after
  the reset-sharding contract was strengthened; all eight wrappers recorded
  postflight status 143, all locks were removed, and every GPU returned to
  4 MiB / 0% with no compute contexts. Partial b03 data is retained only as
  attempt evidence and is not mixed into a later attempt.
- Commit `77dd5da27d9c1a17f7c3cdf8acc712dc6ba13b6a` makes the shard merger reject
  missing or malformed shard receipts, reset duplicates/gaps/order changes,
  reset-manifest drift, candidate duplicates/gaps, and any mode other than
  exact full-pool plus planner-Pareto. The complete source bundle SHA-256 is
  `8e0530add65e65516bf2e9aca631fc51982f20b79fe8c5a2c63d3b9b23377145`.
- `b04`: the first 8-shard resource probe failed closed because each process
  created 17–24 threads inside an eight-CPU lane cpuset. It was stopped before
  any scientific reset completed; all eight official wrappers again recorded
  status 143 and exact empty postflight.
- `b05`: uses eight reset shards per lane with `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
  `VECLIB_MAXIMUM_THREADS` fixed to one. Post-start measured exactly 64 live
  shard processes, three threads per process, eight non-overlapping lane
  cpusets, exactly eight compute contexts per GPU, and less than 10 GiB used on
  every 80 GiB GPU. Evaluation is in progress; final task and release results
  must be appended here before RLD2 is considered closed.

## Release result

Pending completion of b05, per-task independent audits, the unified production
release audit, and exact eight-GPU postflight.

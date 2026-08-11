基于 Dynamic Benchmark 的强化学习
==================================

.. figure:: /_static/svg/dynamic-benchmark.svg
   :align: center
   :width: 90%

   Dynamic Benchmark 任务族与 privileged-state expert 工作流。

Dynamic Benchmark 是面向 Franka 动态操作研究的 MuJoCo 套件。RLinf 从当前 privileged
state 与在线 planner 示教训练轻量 expert policy，再导出经审计的轨迹，供后续
world-action modeling 使用。

概览
----

在 14 个动态抓取、放置、交互与重规划任务上训练可恢复的 BC、SAC、RLPD、
planner-residual RLPD 或 PPO expert。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP actor · 10-head REDQ critic

   .. grid-item-card:: 算法
      :text-align: center

      BC · SAC · RLPD-SAC · residual RLPD · PPO

   .. grid-item-card:: 任务
      :text-align: center

      T1–T5 · P0-Grasp · 共 14 个

   .. grid-item-card:: 硬件
      :text-align: center

      每个独立 run 使用 1 张 GPU

| **你将完成：** 安装私有 benchmark 源码 → 运行 T2 RLPD 配方 → 检查成功与安全门。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 带 ``benchmark`` extra 的本地 SE3-WAM checkout。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 28 58

   * - 任务族
     - Config ID
     - 重点
   * - T1
     - ``t1_belt``, ``t1_xyz``, ``t1_so3``, ``t1_occ``
     - 面向运动、位姿变化、旋转和遮挡的传送带抓取。
   * - T2
     - ``t2_trans``, ``t2_se3``
     - 抓取并放置到平移或完整 SE(3) 运动目标中。
   * - T3
     - ``t3_phase``, ``t3_full``
     - 在 phase-visible 与完整动态条件下执行时序交互。
   * - T4
     - ``t4_sphere``, ``t4_slider``, ``t4_can``, ``t4_sphere_tabletop``
     - 对接触敏感的释放、捕获与物体交互。
   * - T5
     - ``t5_replan``
     - 观测动态事件后在线重规划，不泄漏未来状态。
   * - P0
     - ``p0_grasp``
     - 动态抓取，要求双侧接触、有效保持、离台与稳定抬升。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 说明
   * - 观测 (Observation)
     - 173 维当前 privileged state：138 维数值与 35 维可用性 mask。state-policy 训练关闭相机观测。
   * - 动作 (Action)
     - 归一化 7 维 E7 指令：3 维平移增量、3 维旋转增量和夹爪控制。
   * - 奖励 (Reward)
     - 可独立复算的势函数进展、成功、安全、超时、步惩罚和动作能耗项；schema 为 ``rlinf-dynamic-benchmark-reward-v0.2``。
   * - 提示 (Prompt)
     - 无。task ID 与当前状态共同定义 expert-policy 输入。

安装
----

.. include:: _setup_common.rst

将 ``SE3_WAM_PATH`` 指向已授权的本地 checkout，再安装 environment-only 依赖组合。
源码缺失时安装器会 fail closed。

.. code-block:: bash

   export SE3_WAM_PATH=/path/to/SE3-WAM
   bash requirements/install.sh embodied --env dynamic_benchmark
   source .venv/bin/activate

``embodied-dynamic_benchmark`` Docker target 只安装公开运行时依赖。按
``docker/README.md`` 挂载并安装 research-only SE3-WAM checkout。

运行
----

使用显式源码 identity 与输出目录启动冻结的 T2 RLPD 配方：

.. code-block:: bash

   export SE3_WAM_PATH=/path/to/SE3-WAM
   RLINF_COMMIT=$(git rev-parse HEAD)
   BENCHMARK_COMMIT=$(git -C "$SE3_WAM_PATH" rev-parse HEAD)
   python examples/embodiment/train_dynamic_benchmark_expert.py \
      --config examples/embodiment/config/dynamic_benchmark_t2_rlpd.yaml \
      --rlinf-commit "$RLINF_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --output outputs/dynamic_benchmark/t2_rlpd_seed1

**这个命令会：**

1. 在线收集仅成功的 planner 示教，不读取外部数据集。
2. 先做 BC warm-start，再以等比例 demo/online replay 运行 RLPD-SAC。
3. 在冻结 validation manifest 上评测，并保存模型、replay、环境、优化器、normalizer 与 RNG 状态。

可选参数 ``--actor-bc-weight`` 会在每次在线 actor 更新中加入示教行为克隆损失。
默认值为 ``0``，因此参考 RLPD 配方仍是无正则基线；非零权重必须登记为独立实验臂。

可选参数 ``--reward-safety-penalty`` 控制训练时安全失败的终止奖励，默认值为 ``-10``。
该值必须有限且非正，并写入 run 与示教 cache identity；每个非默认值都必须登记为独立、
仅基于 validation 选择的实验臂。正式评测仍使用 canonical reward 合同，以保证不同策略的
return 可比较。

设置 ``--algorithm residual_rlpd`` 后，实际动作是
``clamp(planner_action + residual_scale * policy_residual, -1, 1)``。replay 与 critic
位于 residual action space，planner 示教严格映射到零 residual；checkpoint 还会保存
有状态的 planner 实例。``--residual-scale`` 默认 ``0.25``，每个 scale 与 actor-BC
权重组合都必须登记为独立实验臂。

精确 CPU process recipe 现在默认开启。没有 YAML 覆盖时，训练和 validation 均使用 32 个
环境；省略 ``--env-worker-processes`` 或 ``--eval-worker-processes`` 时，进程数自动等于
对应 vector width。Linux 默认使用 ``forkserver``，其他平台保留可移植的 ``spawn``。
只要 evaluation process 已开启，默认复用 checkpoint-rewind evaluation worker；
residual-RLPD 还默认在环境所属子进程内计算 privileged planner。需要显式串行兼容模式时，
使用自动生成的 ``--no-persistent-eval-workers``、``--no-eval-planner-in-processes``，或将
两类 process 数都设为 ``0``。sampler/learner overlap 仍默认关闭，因为它会改变 replay 顺序。

manifest row 仍由主进程分配，返回值按 env index 恢复，因此 seed 与 episode 顺序不依赖
worker 完成顺序。每个子进程内部保持串行，对应的 ``--*-worker-threads`` 必须为 ``1``。
worker 继承启动器的 CPU affinity；已测量的 W32 recipe 要求显式限定在同一 NUMA 节点的
32 个逻辑 CPU（例如使用 ``taskset``），process tree 峰值 RSS 约 47 GiB。trainer 无法安全
推断哪一个 NUMA 节点与所用 GPU 相邻。process topology 属于执行 provenance，checkpoint
恢复时必须使用相同配置。使用当前 RLinf 与 SE3-WAM 源码时，task-only manifest、mutable
``MjModel`` 精确 reset 恢复、v0.3 checkpoint 和有界 process cleanup 无需额外开关。

吞吐 bakeoff 前，应在真实 benchmark checkout 上运行进程正确性 gate。它会比较
serial/process 的 reset 与 step digest，验证 process 模式 checkpoint/resume 的精确
后缀，并故意使一个 worker 崩溃，以检查所有分片都能有界清理：

.. code-block:: bash

   python examples/embodiment/verify_dynamic_benchmark_process_runtime.py \
      --task t4_sphere --num-envs 8 --worker-processes 4 \
      --output outputs/dynamic_benchmark/process_gate.json

matched on-policy 对照使用独立的可恢复 trainer：

.. code-block:: bash

   python examples/embodiment/train_dynamic_benchmark_ppo.py \
      --config examples/embodiment/config/dynamic_benchmark_t1_ppo.yaml \
      --rlinf-commit "$RLINF_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --output outputs/dynamic_benchmark/t1_ppo_seed1

该实现使用 7 维 squashed-Gaussian actor、GAE、clipped PPO、value head、冻结
validation manifest 与 update-boundary checkpoint/resume。time-limit truncation 会对
value target 做 bootstrap，但会在 reset 边界停止 advantage 递推。

GPU-native RLPD 可在在线训练前追加
``--demo-policy privileged_teacher --demo-cohorts N
--minimum-demo-success-rate P``，对示教质量做预声明门禁。每个 teacher 动作只读取对应
仍活跃 ``mjwarp_gpu_v1`` lane 的当前状态；trainer 会先原子写出
``demo_quality.json``，若成功率未达到阈值则在任何在线更新前 fail closed。GPU→host 的
观测/终态 mask 读取与 host→GPU 的 teacher 动作传输会单独计入 demonstration control
plane；learned-policy rollout 以及 replay/update 热路径仍保持 device-only。生成的 v0.2
checkpoint 可交给 tensor evaluator，但只有独立 held-out evaluation 通过后才能声称策略
质量达标。

新的 BC/RLPD run 在 planner 收集后写出 ``demo_replay.pt``。matched 算法或正则臂可通过
``--demo-replay-in`` 复用该示教；加载时会严格核对源码 commit、task、state schema、seed、
manifest、环境数和示教合同，并恢复 replay sampling state、normalizer 与收集后的 RNG，
任一 identity 不一致都会 fail closed。

.. warning::

   这些 expert 使用 privileged simulator state。其轨迹是 teacher data，不是可部署
   vision policy 的结果。``--rlinf-commit`` 与 ``--benchmark-commit`` 必须使用完整
   40 位哈希；源码 identity 漂移时 resume 会拒绝运行。

RLD2-QA 校准与正式评测
~~~~~~~~~~~~~~~~~~~~~~

是否传入冻结阈值参数决定正式评测 schema。同时提供 ``--quality-v2-thresholds`` 与
``--expected-quality-v2-thresholds-sha256`` 时，expert 输出
``rlinf-dynamic-benchmark-expert-evaluation-v0.3``，planner 输出
``rlinf-dynamic-benchmark-planner-evaluation-v0.2``；每条 record 都绑定 canonical
``rlinf-dynamic-benchmark-optimal-attempt-v0.3`` tape。仅供 planner 使用的
``metric_calibration`` 波次会刻意省略这两个参数并保持
``rlinf-dynamic-benchmark-planner-evaluation-v0.1``；它是独立科学分区，不是生产
policy 对比。

冻结 Qv3 阈值前，从干净 evaluator checkout 运行 exact-14 × exact-20 planner 波次。
scheduler 必须先完成资源冲突检查，再向 launcher 交付 1 至 8 个 GPU index：

.. code-block:: bash

   RUNTIME_ROOT=/path/to/runtime
   RUNS_ROOT=/path/to/runs
   export RLD2_QA_BENCHMARK_SOURCE_ROOT="$RUNTIME_ROOT/SE3-WAM"
   bash examples/embodiment/rld2_qa_planner_calibration.sh \
      rld2qa-cal 0,1,2,3 "$PWD" "$RUNTIME_ROOT" \
      "$RUNS_ROOT" /path/to/tmp

launcher 会在任何 rollout 前预声明全部 280 个 reset identity，按 canonical 14-task
顺序为每个任务评测 20 行，并在三个不同的 fresh environment 中验证所选的安全成功
planner 轨迹。使用默认 wave ID 时，canonical receipt 写到
``$RUNS_ROOT/RLD2-QA/planner-calibration-metric-v03-s20261350/wave_receipt.json``。
冻结的 ``se3-wam-trajectory-quality-v2-thresholds-v0.3`` 合同必须绑定这份 exact
receipt，并与 validation、review、test-ID 和 test-OOD manifest 保持不相交。

阈值合同冻结后，用独立标识的 evaluator commit 在配对的 20-row validation manifest
上评测 checkpoint：

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

evaluator 会重建 BC/SAC/RLPD、residual-RLPD 与 PPO 策略，核验 checkpoint/源码
identity，保存 reset manifest 和实际动作；每个 rollout 和 replay 都使用独立新建且只
reset 一次的 raw environment，并要求每个 episode 的 action replay 精确通过。正式输出会
递归封存 attempt tape、Qv3 summary/gate、replay receipt 与源码哈希。T5-Replan tape 还会
保存 canonical issued/applied action history，以及从 impact 结束到首次合格 applied
correction 的 causal latency。必须先冻结基于 validation 的策略和超参数选择，之后才能读取
test manifest。

在相同配对 manifest 与阈值 identity 上独立评测 privileged planner，不加载 learned
policy：

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

planner evaluator 同样保存 reset identity、实际动作、exact replay 证据、任务指标和决策
延迟门。对每个 manifest row，planner rollout 与 action-tape replay 都各自在独立新建且
只 reset 一次的 raw environment 上运行；重放 reset 还会先恢复任务隐藏运行态（例如
T5 event tape），再执行 observation、outcome 与 final state 的 exact 检查。上面的带阈值
命令输出 planner v0.2；planner v0.1 仅限 exact-14 calibration launcher 使用。
test-ID/OOD 只用于冻结后的单次比较。

导出 best-known 轨迹
-------------------

首先从 ``rlinf-dynamic-benchmark-optimal-candidates-v0.1`` review candidate manifest
生成配对 Owner-review 子集；其 ``review_contract`` 必须为
``full_generation=false``，planner 必须是 candidate 0，learned candidate 必须携带通过的
v0.2 promotion receipt：

``build_dynamic_benchmark_rld2_promotion.py`` 为每个 v0.2 promotion 构建 receipt 时，
除冻结 threshold path/SHA 与其他必需输入外，还必须通过
``--quality-v2-calibration-wave-receipt`` 和
``--expected-quality-v2-calibration-wave-receipt-sha256`` 接收同一 calibration sidecar。
它会重新打开 sidecar，并把 file/payload identity 与 dataset-relative binding 同时封入
selection evidence 和 promotion receipt。review exporter 要求并重新验证同一参数对。

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

review exporter 输出 ``rlinf-dynamic-benchmark-rld2-paired-review-v0.2`` schema，以
deterministic 方式评估完整 candidate pool，并为每个任务输出六类配对样本。其 receipt 始终
记录 ``full_generation=false``；入选 review 不等于轨迹可用。

.. warning::

   Owner 对每个任务的配对 review 给出明确批准前，禁止启动 accepted-100 生产生成或 full
   release build。不存在可授予或替代该批准的 review CLI 参数。

Owner 批准后，生产输入是 exact-14
``rlinf-dynamic-benchmark-rld2-candidate-release-v0.2``。每个任务的 manifest schema 为
``rlinf-dynamic-benchmark-optimal-candidates-v0.2``，并位于 canonical
``<release-root>/<task>/candidate_manifest.json``；candidate 0 是 planner，所有 candidate
都带 hash-pinned provenance。生产选择必须使用完整 pool 与 ``planner-pareto``：

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

冻结阈值合同会按任务动态派生 10 或 11 个 Qv3 配对维度，而不是硬编码一套全局 phase inventory。
learned attempt 只有满足以下全部条件，才可替换同 reset planner：通过 success、安全、exact
replay、Qv3 absolute gate，以及适用时的 T5 causal gate；在所有冻结 task/Qv3/时长/
控制/能耗/causal 维度上 non-worse；并至少在一个维度上达到 strict improvement。exact tie
由 planner 获胜，return 仅用于诊断。

exporter 同时要求冻结 threshold path/SHA 与权威 calibration receipt source path/SHA。它会
重新打开 canonical exact-14 × exact-20 receipt，与 v0.3 阈值绑定逐项核对 task/reset
identity，再按合同记录的安全 dataset-relative ``provenance/.../wave_receipt.json`` 路径将其
复制进数据集。

中断后可用相同命令追加 ``--resume``。恢复前会严格核验源码/候选 identity，把未提交尾部保存在
同级 recovery 目录，将 JSONL 截断到最后一次原子提交的 reset 边界，并重跑该 reset。每次
attempt 都保存轻量 state/action/reward tape 与 exact-replay 证据；winner 额外保存 RGB-D/HDF5。

如果独立选出的轻量 winner 未通过 canonical render replay，该 reset 会被拒绝而不是发布。新导出
会把结构化 ``render_parity_skip`` 证据绑定到被选 attempt，并在 sealed recovery log 中保留
失败事件。分片导出必须使用 ``merge_optimal_export_shards.py`` 封存，保证这些事件在前缀截断后
仍被保留。auditor 只有在独立选出同一 attempt、确认没有发布 winner、并校验匹配的 recovery
证据后才接受 skip；skip reset 永远不计入 accepted 配额。

消费数据集前必须运行独立 auditor。计算最终 ``dataset_card.json`` 与根目录
``SHA256SUMS`` 的哈希，并同时传入冻结的 Qv3 threshold identity：

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

auditor 会独立复算根目录 checksum、tape shape/hash、score、full-pool Pareto 选择、T5
issued/applied causal 证据、benchmark exact replay 与 HDF5/轻量 action parity；还会重新打开
dataset-local calibration receipt，核对 canonical bytes、dataset-relative 路径与 threshold
SHA 绑定。只有全部通过才写入 ``training_eligible=true``。这里的 ``optimal`` 指冻结
candidate/reset/budget 合同内的 best-known，不代表连续控制全局最优证明。

可视化与结果
------------

训练期间读取 ``metrics.jsonl``，完成后读取 ``summary.json``。策略按成功、安全、完成度、
return、时长和动作能耗执行词典序排名。run 还会写出 ``best_policy.pt``、
``final_policy.pt`` 与 ``checkpoint_latest.pt``。若调度验证已经对应最后一个环境 step，trainer
会复用它并记录 ``validation_reused``，不再重复相同评测。多种子 benchmark 筛选完成前不发布参考成功率。

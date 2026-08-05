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

设置 ``--algorithm residual_rlpd`` 后，实际动作是
``clamp(planner_action + residual_scale * policy_residual, -1, 1)``。replay 与 critic
位于 residual action space，planner 示教严格映射到零 residual；checkpoint 还会保存
有状态的 planner 实例。``--residual-scale`` 默认 ``0.25``，每个 scale 与 actor-BC
权重组合都必须登记为独立实验臂。

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

新的 BC/RLPD run 在 planner 收集后写出 ``demo_replay.pt``。matched 算法或正则臂可通过
``--demo-replay-in`` 复用该示教；加载时会严格核对源码 commit、task、state schema、seed、
manifest、环境数和示教合同，并恢复 replay sampling state、normalizer 与收集后的 RNG，
任一 identity 不一致都会 fail closed。

.. warning::

   这些 expert 使用 privileged simulator state。其轨迹是 teacher data，不是可部署
   vision policy 的结果。``--rlinf-commit`` 与 ``--benchmark-commit`` 必须使用完整
   40 位哈希；源码 identity 漂移时 resume 会拒绝运行。

冻结 validation 选择后，可用独立标识的 evaluator commit 在确定性的 test-ID 或
test-OOD manifest 上评测 best checkpoint：

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

evaluator 会重建 BC/SAC/RLPD、residual-RLPD 与 PPO 策略，核验 checkpoint/源码
identity，保存 reset manifest 和实际动作，并要求每个 episode 的 action replay 精确通过；
结果包含确定性的成功率、安全失败、完成度、动作能耗与决策延迟。必须先冻结基于 validation
的策略和超参数选择，之后才能读取 test manifest；冻结前的 evaluator 工程 smoke 使用
``--split validation``。

冻结前用相同 paired validation manifest 独立评测 privileged planner，不加载 learned
policy：

.. code-block:: bash

   python examples/embodiment/evaluate_dynamic_benchmark_planner.py \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --benchmark-commit "$BENCHMARK_COMMIT" \
      --task t1_xyz \
      --split validation \
      --manifest-seed 20261150 \
      --episodes 8 \
      --output outputs/dynamic_benchmark/t1_xyz_planner/validation_seed1

planner evaluator 同样保存 reset identity、实际动作、exact replay 证据、任务指标和决策
延迟门；test-ID/OOD 只用于冻结后的单次比较。

导出 best-known 轨迹
-------------------

基于 validation 冻结候选池后，使用
``rlinf-dynamic-benchmark-optimal-candidates-v0.1`` JSON manifest：candidate index 0
必须是唯一 planner，其后是 hash-pinned policies。每个 policy 条目记录路径、SHA-256、
stochastic 标志、exploration seed offset 与可选 residual-scale override。exporter 使用冻结的
8→16→32 候选升级预算，并按 success、安全、完成度、return、控制步数、动作能耗的稳定
词典序选择 winner：

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

中断后可用相同命令追加 ``--resume``。恢复前会严格核验源码/候选 identity，把未提交尾部保存在
同级 recovery 目录，将 JSONL 截断到最后一次原子提交的 reset 边界，并重跑该 reset。每次
attempt 都保存轻量 state/action/reward tape 与 exact-replay 证据；winner 额外保存 RGB-D/HDF5。

如果独立选出的轻量 winner 未通过 canonical render replay，该 reset 会被拒绝而不是发布。新导出
会把结构化 ``render_parity_skip`` 证据绑定到被选 attempt，并在 sealed recovery log 中保留
失败事件。分片导出必须使用 ``merge_optimal_export_shards.py`` 封存，保证这些事件在前缀截断后
仍被保留。auditor 只有在独立选出同一 attempt、确认没有发布 winner、并校验匹配的 recovery
证据后才接受 skip；skip reset 永远不计入 accepted 配额。

消费数据集前必须运行独立 auditor，并传入 exporter 最终打印的 ``dataset_card.json`` 与
``checksums.sha256`` 哈希：

.. code-block:: bash

   python examples/embodiment/audit_dynamic_benchmark_optimal_trajectories.py \
      --dataset-root outputs/dynamic_benchmark/t2_optimal_v1 \
      --expected-dataset-card-sha256 "$DATASET_CARD_SHA" \
      --expected-checksums-sha256 "$CHECKSUMS_SHA" \
      --expected-candidate-manifest-sha256 "$CANDIDATES_SHA" \
      --auditor-commit "$EVALUATOR_COMMIT" \
      --output outputs/dynamic_benchmark/t2_optimal_v1.audit.json

auditor 会独立复算根目录 checksum、tape shape/hash、score、升级预算、winner 选择、benchmark
exact replay 与 HDF5/轻量 action parity。只有全部通过才写入 ``training_eligible=true``。
这里的 ``optimal`` 指冻结 candidate/reset/budget 合同内的 best-known，不代表连续控制全局最优证明。

可视化与结果
------------

训练期间读取 ``metrics.jsonl``，完成后读取 ``summary.json``。策略按成功、安全、完成度、
return、时长和动作能耗执行词典序排名。run 还会写出 ``best_policy.pt``、
``final_policy.pt`` 与 ``checkpoint_latest.pt``。若调度验证已经对应最后一个环境 step，trainer
会复用它并记录 ``validation_reused``，不再重复相同评测。多种子 benchmark 筛选完成前不发布参考成功率。

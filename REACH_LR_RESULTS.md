# 平面左右伸手实验 · 2026-11-19

## 评估口径

`end_effector` 的位置使用 MuJoCo 世界坐标，扣除 Warp 并行环境的原点平移。模型起始朝向下 +X 向前、+Y 向左、+Z 向上。IK 指定点以示范预站立后的手爪位置为原点设置。曲线、方向完成率和收回残差以各轨迹第 0 帧为位移零点。绝对位置误差直接在共同世界坐标中计算。

所有控制器从参考第 0 帧的物理状态开始。50 Hz 取样，完整结果为 150 帧，帧时标 0–2.98 s。伸出指令约在 0.25–1.10 s，1.75–2.60 s 收回。停留段取闭区间 [1.45, 1.70] s，对应帧 73–85。设停留段平均位移为 `d_actual`，参考停留段平均位移为 `d_ref`。方向完成率是 `100 × dot(d_actual, d_ref) / dot(d_ref, d_ref)`，可超过 100%。参考误差按逐帧欧氏距离取均值，IK 误差按同一批帧到固定指定点的距离取均值。收回残差是最后一帧相对起点的位移模长。机身倾角是全程机身 Z 轴相对世界 Z 轴的最大夹角。CPU 示范逐帧记录参与接触的脚部碰撞几何，Warp 评估逐帧记录各脚与地面的接触。

验收要求为 150 帧、四脚支撑、最大倾角 ≤5°、停留方向完成率 ≥70%、停留段平均参考误差 ≤2.5 cm。另列全程平均参考误差和 IK 指定点误差。

## 原有两条留出基线

以下结果均为 150/150 帧，四脚支撑 100%。CPU IK＋PD 从原始偏移重新生成，重算的参考误差四舍五入为零。两条基线都有 [左侧 XYZ 及产物](artifacts/reach_lr_20261119/baseline/reach_00/displacement_xyz.png) 和 [右侧 XYZ 及产物](artifacts/reach_lr_20261119/baseline/reach_01/displacement_xyz.png)。

| 原留出点 | 控制器 | 完成率 | 停留参考误差 cm | 全程参考误差 cm | IK 点误差 cm | 最大倾角 ° | 收回残差 cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | CPU IK＋PD | 100.00% | 0.000 | 0.000 | 2.901 | 2.401 | 0.822 |
| 00 | Warp IK＋PD | 100.04% | 0.004 | 0.005 | 2.903 | 2.405 | 0.835 |
| 00 | 旧纯初始化 | 15.29% | 5.847 | 3.111 | 5.987 | 2.450 | 1.507 |
| 00 | 旧更新后 | 18.54% | 5.618 | 3.023 | 5.598 | 2.424 | 1.170 |
| 01 | CPU IK＋PD | 100.00% | 0.000 | 0.000 | 2.647 | 2.443 | 0.677 |
| 01 | Warp IK＋PD | 100.04% | 0.003 | 0.004 | 2.649 | 2.444 | 0.680 |
| 01 | 旧纯初始化 | 15.07% | 5.407 | 2.896 | 4.695 | 2.450 | 1.560 |
| 01 | 旧更新后 | 12.94% | 5.601 | 3.081 | 4.677 | 2.425 | 1.205 |

旧更新后策略的停留 Y 位移分别为 +0.013 和 +0.010 cm，参考为 +2.424 和 −2.743 cm。两条参考的峰值位移分别约 7.05 cm 和 6.34 cm，旧更新后策略分别约 1.57 cm 和 1.56 cm。

关节名在 NPZ、MuJoCo 模型、Warp 机器人和动作项中一致。前 12 项是 FL、FR、RL、RR 的 hip、thigh、calf，末 6 项是 `joint1`–`joint6`。每个关节均用 `q_target = q_home + 0.35 × action`，评估编码器偏置为零。臂的默认角为 `(0, 0.6, −1.4, 0, 0.45, 0)` rad。基线复位的关节误差为 0，手爪位置差约 `6.0e-8` m。

复位观测的物理状态是第 0 帧，命令窗口从第 1 帧开始。实测第 `k` 帧的动作由上一个观测决定，其参考窗口以 `k` 开始，执行后进入下一窗口。第 149 帧以末帧封顶。旧初始化的动作标签额外领先一帧。两组新短训都修正这一帧，控制变量比较因此共享相同帧处理。加载旧权重的基线重跑仍按历史检查点的策略和模型配置执行。

固定同一份机器人观测，对换原训练的左/右参考窗口。观测第 20、30、40 帧分别使用参考 21–30、31–40、41–50 帧。下表为第 30 帧探测的右减左差，单位是动作缩放后的 rad。

| 策略 | joint1 | joint2 | joint3 | joint4 | joint5 | joint6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧纯初始化 | −3.42e−5 | −1.24e−5 | −8.89e−6 | −3.06e−5 | 8.85e−6 | −1.28e−7 |
| 旧更新后 | −3.43e−5 | −1.22e−5 | −8.95e−6 | −3.08e−5 | 8.99e−6 | −2.04e−7 |

三次切换中旧策略的单关节绝对变化均小于 `3.5e-5` rad。校准组在训练集上的同类探测中，`joint1` 的单关节变化最高约 `0.017` rad，左右轨迹也实际分开。完整六关节缩放前输出、缩放后差和命令帧保存在每个策略的 `*_audit.json`。

## 改动和训练侧验证

主要变量是示范生成是否补偿实际 PD 物理回放的停留误差。用同一站姿、样条时间律、PD 增益和限力，对修正后的 IK 目标重新跑物理仿真。每次将实际停留段的平均手爪残差加到下一轮 IK 目标，平均残差达到 1.5 mm 或最多尝试八次时结束。保存的 `reach_goal_w` 始终是原始指定点。

原始 IK 与校准两组使用完全相同的六个平面训练目标。校准前 CPU 示范的 IK 点误差为 2.62–3.20 cm，校准后为 0.095–0.123 cm，六条示范均完整、四脚支撑。校准训练目标 `(5, 4, 0)` 和 `(5, −4, 0)` cm 的 Warp 直接指令回放均完成 150 帧，四脚支撑 100%，距原始指定点分别为 0.094、0.096 cm，距 CPU 逐帧参考的停留误差均约 0.003 cm。两条训练目标上，原始 IK 对照的纯初始化策略相对校准参考的停留误差为 3.24 和 2.79 cm，校准组纯初始化为 0.25 和 0.45 cm。以训练侧结果冻结方案，随后生成并运行预先锁定的两个留出点。

两组使用种子 0、48 个并行环境、401 步示范回归、100 次环境采样迭代和 24 次 SuperMimic 更新。世界模型和策略批量均 64，想象视野 12 步，策略更新初始学习率 `1e-5`。两组使用相同的上游跟踪奖励。回放把倾角与四脚着地作为独立测量。

| 新短训 | 整卡起始 / 实测峰值 MiB | PyTorch 峰值已分配 / 保留 MiB | 更新次数 |
| --- | ---: | ---: | ---: |
| `lr_ik_100` | 1132 / 2101 | 363.7 / 450.0 | 24 |
| `lr_calibrated_100` | 815 / 1430 | 363.7 / 450.0 | 24 |

整卡由 `nvidia-smi` 每 200 ms 采样，含 Windows 桌面、CUDA 和 Warp。桌面占用在两个运行之间有变化。PyTorch 的计数只覆盖其分配器。运行环境为 WSL2 Ubuntu 24.04.5 LTS、RTX 4060 Laptop（报告容量 8188 MiB）、驱动 581.80、mjlab 1.6.0、MuJoCo / Warp 3.16.0 和 PyTorch 2.14.1+cu130。

## 两个新留出目标

左目标 `reach_00` 的原始 IK 偏移为 `(5.5, +3.2, 0)` cm，右目标 `reach_01` 为 `(6.2, −3.4, 0)` cm。它们在训练前锁定，在方案冻结后首次运行。所有策略输入相同的校准参考并按它评分。以下各条均完成 150/150 帧，四脚支撑 100%。

| 目标 | 控制器 | 完成率 | 停留参考误差 cm | 全程参考误差 cm | IK 点误差 cm | 最大倾角 ° | 收回残差 cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 左 | 校准 CPU IK＋PD | 100.00% | 0.000 | 0.000 | 0.065 | 2.419 | 0.934 |
| 左 | 校准 Warp IK＋PD | 100.00% | 0.003 | 0.004 | 0.065 | 2.423 | 0.946 |
| 左 | 旧纯初始化 | 14.50% | 5.462 | 2.982 | 5.413 | 2.450 | 1.495 |
| 左 | 旧更新后 | 18.59% | 5.197 | 2.843 | 5.147 | 2.424 | 1.163 |
| 左 | 原始 IK 对照纯初始化 | 87.92% | 2.774 | 1.512 | 2.779 | 2.445 | 0.784 |
| 左 | 原始 IK 对照更新后 | 85.51% | 2.781 | 1.528 | 2.783 | 2.442 | 0.774 |
| 左 | **校准纯初始化** | **79.30%** | **1.344** | **0.990** | **1.288** | **2.367** | **0.242** |
| 左 | 校准更新后 | 75.09% | 1.603 | 1.031 | 1.546 | 2.370 | 0.233 |
| 右 | 校准 CPU IK＋PD | 100.00% | 0.000 | 0.000 | 0.066 | 2.420 | 0.949 |
| 右 | 校准 Warp IK＋PD | 100.00% | 0.003 | 0.005 | 0.067 | 2.425 | 0.962 |
| 右 | 旧纯初始化 | 13.69% | 6.105 | 3.279 | 6.060 | 2.450 | 1.627 |
| 右 | 旧更新后 | 16.71% | 5.894 | 3.179 | 5.848 | 2.424 | 1.252 |
| 右 | 原始 IK 对照纯初始化 | 86.90% | 2.860 | 1.584 | 2.860 | 2.416 | 0.714 |
| 右 | 原始 IK 对照更新后 | 89.47% | 2.834 | 1.588 | 2.838 | 2.419 | 0.714 |
| 右 | **校准纯初始化** | **82.56%** | **1.462** | **1.189** | **1.412** | **2.354** | **0.396** |
| 右 | 校准更新后 | 79.89% | 1.564 | 1.169 | 1.513 | 2.354 | 0.344 |

校准纯初始化是本轮跟踪效果最好的检查点。停留 Y 位移为 +2.80 和 −3.50 cm，收回残差为 0.24 和 0.40 cm。对比同样使用六条平面示范的原始 IK 组，校准使纯初始化的停留参考误差分别减少 1.43 和 1.40 cm。原始 IK 组的完成率已经过线，手爪却在停留时偏低约 2.6 cm，四条结果的参考误差超线 0.27–0.36 cm。旧更新策略在新留出上分别差 51.41 和 53.29 个百分点才达到完成率门槛，停留参考误差分别超线 2.70 和 3.39 cm。

24 次 SuperMimic 更新后，校准组仍通过全部验收项。相对纯初始化，左、右方向完成率各下降 4.21 和 2.67 个百分点，停留参考误差分别增加 0.259 和 0.102 cm。这次改善在纯示范初始化时已经出现。新平面训练集同时改变了原历史基线的示范覆盖，原始 IK 对照和校准组之间的比较控制了这一因素。新结果来自一个种子及两个留出点。

- 左目标。[逐条指标](artifacts/reach_lr_20261119/holdout/reach_00/metrics.json) · [XYZ 图](artifacts/reach_lr_20261119/holdout/reach_00/displacement_xyz.png) · [纯初始化视频](artifacts/reach_lr_20261119/holdout/reach_00/calibrated_initial.mp4) · [更新后视频](artifacts/reach_lr_20261119/holdout/reach_00/calibrated_updated.mp4)
- 右目标。[逐条指标](artifacts/reach_lr_20261119/holdout/reach_01/metrics.json) · [XYZ 图](artifacts/reach_lr_20261119/holdout/reach_01/displacement_xyz.png) · [纯初始化视频](artifacts/reach_lr_20261119/holdout/reach_01/calibrated_initial.mp4) · [更新后视频](artifacts/reach_lr_20261119/holdout/reach_01/calibrated_updated.mp4)

## 独立复现

在原生 Ubuntu 22.04.5、RTX 5090 和 580.126.09 驱动上，从提交 `1115352` 按 README 与下方命令重新运行。使用锁定的 Python 3.12.13、mjlab 1.6.0、MuJoCo / Warp 3.16.0 和 PyTorch 2.14.1+cu130。发布权重回放使用随仓库提供的校准留出参考。重训用原始 IK 和校准两套示范，均覆盖相同的 6 个目标，并使用种子 0、48 个并行环境、401 步示范初始化、100 次环境采样迭代和 24 次 SuperMimic 更新。重训后生成两条校准留出参考，逐帧数据与原实验发布的参考一致。

以下各条均跑满 150 帧，四脚支撑 100%，最大倾角低于 2.5°。验收还要求方向完成率至少 70%，停留参考误差不超过 2.5 cm。

| 策略 | 左完成率 | 左停留参考误差 cm | 右完成率 | 右停留参考误差 cm | 两个目标的验收结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 发布的校准纯初始化权重 | 79.30% | 1.344 | 82.56% | 1.462 | 通过 |
| 原始 IK 纯初始化复训 | 87.84% | 2.775 | 86.81% | 2.860 | 停留误差超限 |
| 原始 IK 更新后复训 | 85.24% | 2.786 | 89.58% | 2.831 | 停留误差超限 |
| 校准纯初始化复训 | 79.58% | 1.321 | 81.99% | 1.482 | 通过 |
| 校准更新后复训 | 75.19% | 1.594 | 79.94% | 1.559 | 通过 |

发布权重的回放与原报告在表中精度下一致。重训时，校准示范让纯初始化策略的左右停留误差分别减少 1.45 和 1.38 cm。继续更新 24 次后，校准组仍达标，左右误差分别增加 0.273 和 0.077 cm。这次复现覆盖一个训练种子和两个仿真留出目标。

历史短训也在同一台机器上重新执行。重新生成的 8 条示范与仓库中原示范的共有数据字段逐项一致。下表的到点误差以 IK 指定点为基准，取伸臂停留段的手爪平均距离。三种策略均跑满两条各 150 帧的轨迹，全程四脚接触地面。

| 历史训练流程 | 迭代 / 更新 | 留出 1 到点误差 cm | 留出 2 到点误差 cm |
| --- | ---: | ---: | ---: |
| 从头短训 | 300 / 104 | 8.46 | 4.23 |
| 仅示范初始化 | 0 / 0 | 5.98 | 4.69 |
| 示范初始化后更新 | 150 / 44 | 5.59 | 4.66 |

示范初始化前后两份结果与 README 的原始记录在厘米小数点后两位一致。从头短训的原记录为 8.64 和 4.71 cm。

- 发布权重回放。[左目标指标](artifacts/repro_20261119/release_replay/reach_00/metrics.json) · [右目标指标](artifacts/repro_20261119/release_replay/reach_01/metrics.json)
- 左目标复训。[逐条指标](artifacts/repro_20261119/left_right/holdout/reach_00/metrics.json) · [XYZ 图](artifacts/repro_20261119/left_right/holdout/reach_00/displacement_xyz.png) · [校准纯初始化视频](artifacts/repro_20261119/left_right/holdout/reach_00/calibrated_initial.mp4)
- 右目标复训。[逐条指标](artifacts/repro_20261119/left_right/holdout/reach_01/metrics.json) · [XYZ 图](artifacts/repro_20261119/left_right/holdout/reach_01/displacement_xyz.png) · [校准纯初始化视频](artifacts/repro_20261119/left_right/holdout/reach_01/calibrated_initial.mp4)

## 复现

在 Ubuntu 的仓库根目录运行，`r` 取一个新的输出根目录。原先的 `data/` 和 `artifacts/training/` 用于读原基线。公开克隆可在 [Release](https://github.com/violet446/go2-piper-supermimic-v2/releases/tag/reach-lr-20261119) 下载 `go2-piper-legacy-initial.pt` 和 `go2-piper-legacy-updated.pt`，分别放到 `artifacts/training/reach_warm_initial/policy.pt` 和 `artifacts/training/reach_warm/ckpt_latest.pt`。

```bash
r=artifacts/reach_lr_repro
mkdir "$r"
bash run.sh reach-lr baseline "$r/baseline" > "$r/baseline.log" 2>&1
bash run.sh reach-lr train-data "$r/train_data"
bash run.sh reach-lr compare "$r/train_replay" --data-dir "$r/train_data/calibrated" \
  --clip 1 --clip 4 --no-video > "$r/train_replay.log" 2>&1
bash run.sh reach-lr-warm lr_ik_100 100 "$r/control_ik" "$r/train_data/ik"
bash run.sh reach-lr-warm lr_calibrated_100 100 "$r/control_calibrated" "$r/train_data/calibrated"
ik="$r/control_ik/training/lr_ik_100"
cal="$r/control_calibrated/training/lr_calibrated_100"
new_checkpoints=(
  --checkpoint ik_initial "${ik}_initial/policy.pt"
  --checkpoint ik_updated "$ik/ckpt_latest.pt"
  --checkpoint calibrated_initial "${cal}_initial/policy.pt"
  --checkpoint calibrated_updated "$cal/ckpt_latest.pt"
)
bash run.sh reach-lr compare "$r/train_controls" --data-dir "$r/train_data/calibrated" \
  --clip 1 --clip 4 --no-video "${new_checkpoints[@]}" \
  --probe-paths "$r/train_data/calibrated/reach_00.npz" "$r/train_data/calibrated/reach_03.npz" \
  > "$r/train_controls.log" 2>&1
bash run.sh reach-lr holdout-data "$r/holdout_data"
bash run.sh reach-lr compare "$r/holdout" --data-dir "$r/holdout_data/calibrated" \
  --probe-paths "$r/train_data/calibrated/reach_01.npz" "$r/train_data/calibrated/reach_04.npz" \
  --checkpoint legacy_initial artifacts/training/reach_warm_initial/policy.pt \
  --checkpoint legacy_updated artifacts/training/reach_warm/ckpt_latest.pt \
  "${new_checkpoints[@]}" > "$r/holdout.log" 2>&1
```

各目标目录包含 `metrics.json`、`displacement_xyz.png`、逐帧 `*_rollout.npz`、动作与帧审计和并排视频。轨迹图灰带标出停留窗口。视频以 25 fps 显示隔帧采样，完整数值评估用原始 50 Hz 帧。

用相同检查点独立复跑了两个校准策略的两条留出轨迹。两次输出的逐帧数据按位一致。20 个对比视频通过解码检查，每个 75 帧。

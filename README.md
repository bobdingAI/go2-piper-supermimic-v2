# Go2 + PiPER 站立伸手

本项目在 WSL2 Ubuntu 24.04 的 Python 3.12 环境运行，使用 RTX 4060 Laptop GPU、mjlab 1.6.0 和 MuJoCo Warp 3.16.0。实施顺序见 [计划](PLAN.md)。这版验证了带臂仿真、动作数据生成、SuperMimic 更新和留出轨迹评估。

2026-11-19 的平面左右伸手实验记录了统一评估口径、实测结果和独立运行命令。见 [左右伸手实验](REACH_LR_RESULTS.md)。

- [示范与策略对比视频（下载 MP4）](https://github.com/violet446/go2-piper-supermimic-v2/releases/download/reach-lr-20261119/go2_piper_demo_vs_policy.mp4) 左栏是示范，右栏是本轮仅用示范训练的策略，依次播放两个留出目标。
- [评估用策略权重与视频下载](https://github.com/violet446/go2-piper-supermimic-v2/releases/tag/reach-lr-20261119)

策略每个控制周期同时输出 12 个腿关节和 6 个臂关节的指令。示范的腿关节目标设为固定站姿，臂关节目标随伸手动作变化，夹爪开合固定。

## 本机结果

在自由浮动的 Go2 上启用 12 个腿关节和 6 个 PiPER 臂关节，手爪保持固定。用 IK 生成关节指令，在带地面的 MuJoCo 物理仿真里用 PD 执行，再以 50 Hz 记录实际运动。6 条用于训练，2 条按不同伸手终点留作评估。8 条参考都全程保持四脚着地，最大机身倾角为 2.45°。

| 模型 | SuperMimic 迭代 / 更新 | 留出 1 到点误差 | 留出 2 到点误差 | 最大倾角 |
|---|---:|---:|---:|---:|
| 从头短训练 | 300 / 104 | 8.64 cm | 4.71 cm | 4.30° |
| 仅示范初始化 | 0 / 0 | 5.98 cm | 4.69 cm | 2.45° |
| 示范初始化后做 SuperMimic 更新 | 150 / 44 | 5.59 cm | 4.66 cm | 2.42° |

每个策略都完整跑完两条 3 秒留出轨迹。到点误差是伸臂停留段的手爪位置与 IK 请求点的距离。参考本身经 PD 执行后到请求点也还差约 2.9 和 2.6 cm。短训练足以验证更新和回放，现有试验只涵盖站立状态下少量近距离目标。示范初始化后继续做 44 次更新带来的增益很小。

从头实验用 48 个并行环境、两个 64 样本的训练批次、12 步想象视野。在训练期间做的 `nvidia-smi` 采样，整张卡峰值为 1,745 MiB，含桌面和同时运行的评估。上游裸 Rover 的 72 GiB 和 20,480 个环境是完整实验配置。

## 文件与复现

- `vendor/go2_piper/` 是固定版本的组合模型。源 XML 保留，`reference.py` 生成供本项目使用的派生 XML。派生版本改用 mjlab 的执行器、默认接触选项和零碰撞 margin，以适配 MuJoCo Warp。
- `vendor/supermimic-quadruped-0.2.0/` 包含四足 SuperMimic 和本项目添加的 Go2 + PiPER 机器人配置。
- `data/train/` 和 `data/holdout/` 保存参考轨迹。保存在轨迹内的示范关节目标用于可选的初始化阶段。
- `artifacts/` 下有 GPU 采样、评估数据和视频。本机运行还会生成原始日志、TensorBoard 记录和完整续训检查点。GitHub Release 提供可直接回放的策略权重和配置。对比视频左侧是 IK 与 PD 生成的参考，右侧是策略在物理仿真中的表现。

在 WSL Ubuntu 24.04 中运行。项目使用 Python 3.12、CUDA 和支持 EGL 的 MuJoCo 渲染环境。新克隆可用 `uv` 根据随附的锁文件创建环境。

```bash
runtime="$HOME/.cache/go2-piper-supermimic-v2"
UV_PROJECT_ENVIRONMENT="$runtime/venv" uv sync --locked --no-dev --python 3.12 \
  --project vendor/supermimic-quadruped-0.2.0
uv pip install --python "$runtime/venv/bin/python" imageio-ffmpeg==0.6.0 matplotlib==3.11.2
```

在公开仓库中评估本轮已训练策略时，先下载示范初始化检查点。以下命令使用随仓库发布的两个校准留出参考。

```bash
model_dir=artifacts/reach_lr_20261119/control_calibrated/training/lr_calibrated_100_initial
mkdir -p "$model_dir"
curl -fL https://github.com/violet446/go2-piper-supermimic-v2/releases/download/reach-lr-20261119/go2-piper-calibrated-initial.pt \
  -o "$model_dir/policy.pt"
bash run.sh reach-lr compare artifacts/reach_lr_replay \
  --data-dir artifacts/reach_lr_20261119/holdout_data/calibrated \
  --checkpoint calibrated_initial "$model_dir/policy.pt"
```

从头重跑历史短训的命令如下。已有本机实验可用新运行名保存新结果。本轮完整对照的独立运行命令见 [实验报告](REACH_LR_RESULTS.md#复现)。

```bash
bash run.sh reference
bash run.sh train reach_smoke 300
bash run.sh warm reach_warm 150
bash run.sh evaluate --checkpoint artifacts/training/reach_warm/ckpt_latest.pt --clip 0
bash run.sh evaluate --checkpoint artifacts/training/reach_warm/ckpt_latest.pt --clip 1 --no-video
```

机器人资产来自 [LeggedManip Lab 的 Go2 + PiPER 模型](https://github.com/zzzJie-Robot/LeggedManip_Lab/tree/4e12109d1eccb1d9da2e4a35f189fabea3f13677/mujoco/robots/go2_piper)。

`vendor/go2_piper/training_model.xml` 为项目修改的派生模型。随附 [上游机器人资产许可证](vendor/go2_piper/LICENSE) 和 [SuperMimic 第三方说明](vendor/supermimic-quadruped-0.2.0/THIRD_PARTY_NOTICES.md)。

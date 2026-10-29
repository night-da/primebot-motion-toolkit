# Episode 生成实现

本目录保存动作来源、轨迹和仿真实现。普通使用者应从父目录调用
`generate_episode.py`，不需要直接运行本目录中的模块。

```text
CSV 或 motions.py
→ trajectory.py 计算每个时刻的目标角
→ simulation.py 执行 PD 控制和 MuJoCo 步进
→ 父目录 generate_episode.py 记录 transition 并输出 NPZ/JSON/MP4
```

文件职责：

- 父目录 `generate_episode.py`：仿真循环、视频记录与三个结果文件的输出。
- `simulation.py`：Q1 控制命令、PD 力矩、重力补偿和仿真步进。
- `motions.py`：CSV 读取和代码规划的挥手动作。
- `trajectory.py`：移动、保持、正弦摆动和轨迹插值。
- `examples/`：敬礼 CSV 示例。

共享的 22 关节定义和 MuJoCo 映射位于父目录的 `common/`。

输入动作使用弧度。CSV 可以包含 Q1 的部分或全部关节；未出现的关节使用默认姿态。
仿真固定以 `1000 Hz` 记录完整的 22 关节 transition，视频单独按 `60 FPS`
采样。

每个仿真步记录：

```text
observation[t] -- action[t] --> next_observation[t+1]
```

最终输出为 `control_transition.npz`、`control_transition.json` 和
`preview.mp4`。完整安装方式、CSV 规范和命令见父目录 `README.md`。

# PrimeBot Q1 动作仿真与数据工具包

本项目用于完成以下流程：

```text
CSV 或代码规划的动作
→ MuJoCo 固定基座仿真
→ NPZ/JSON/MP4 Episode
→ LeRobot v3 数据集
```

## 1 目录结构

```text
primebot-motion-toolkit/
├── common/                             # Q1 关节定义和 MuJoCo 映射
├── dataset/
│   ├── episodes/                       # 两个可直接查看的 Episode 示例
│   └── lerobot/                        # 两个 LeRobot v3 数据集示例
├── episode_generator/
│   ├── examples/                       # 敬礼 CSV 示例
│   ├── motions.py                      # CSV 读取和代码规划动作
│   ├── simulation.py                   # MuJoCo 控制与仿真
│   └── trajectory.py                   # 轨迹插值
├── model/
│   ├── xml/q1_v3_mc.xml                # 默认 Q1 MuJoCo 模型
│   └── meshes/                         # 模型所需的 STL 网格
├── generate_episode.py                 # 生成 Episode
├── export_lerobot_dataset.py           # 转换为 LeRobot v3
└── joint_pose_editor.py                # 关节角度编辑器
```

模型来源：<https://atomgit.com/primebot/q1-urdf.git>

## 2 安装运行环境

推荐在 Miniforge Prompt 中创建 Python 3.12 环境：

```powershell
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg
python -m pip install "lerobot[dataset]==0.6.1"
python -m pip install "mujoco==3.17.0"
```

检查安装：

```powershell
python -c "import mujoco, numpy, cv2, lerobot; print('环境正常')"
ffmpeg -version
```

说明：

- 生成 Episode 需要 MuJoCo、NumPy 和 FFmpeg。
- 转换 LeRobot 数据集还需要 OpenCV 和 LeRobot。
- 关节编辑器使用 Python 自带的 Tkinter，并会同时打开 MuJoCo Viewer。

## 3 输入动作 CSV

CSV 第一行是关节名称，后续每行是该时刻的目标关节角，单位为弧度。例如：

```csv
left_shoulder_pitch_joint,left_shoulder_roll_joint,left_shoulder_yaw_joint,left_elbow_joint
0.0,0.18,0.0,1.4
0.0,0.19,0.01,1.38
```

要求：

- 关节名必须存在于 Q1 模型中，且不能重复。
- 数值必须有限并处于模型关节范围内。
- CSV 可以只包含本动作使用的部分关节；其他关节使用默认姿态。
- 没有时间列时，通过 `--csv-fps` 指定采样频率。
- 第一列也可以是 `t`、`time`、`timestamp` 或 `timestamp_s`；此时直接使用文件中的时间，忽略 `--csv-fps`。

项目已提供敬礼示例：

```text
episode_generator/examples/q1_salute_reference_120fps.csv
```

## 4 生成 Episode

### 4.1 从敬礼 CSV 生成

```powershell
python .\generate_episode.py `
  --from-csv .\episode_generator\examples\q1_salute_reference_120fps.csv `
  --csv-fps 120 `
  --output .\output\episodes\q1_salute\episode_000
```

### 4.2 从代码规划的挥手动作生成

```powershell
python .\generate_episode.py `
  --motion wave `
  --output .\output\episodes\q1_wave\episode_000
```

输出目录必须不存在，防止误覆盖已有数据。程序默认加载包内的
`model/xml/q1_v3_mc.xml`，需要测试其他 MJCF 时可以使用 `--model` 覆盖。

当前生成过程固定采用：

- 固定基座的试验台模式；
- `1000 Hz` MuJoCo 控制与状态记录；
- `800 × 600、60 FPS` 视频。

固定基座适合观察和记录动作，不代表机器人具备自由站立或动态平衡能力。

## 5 Episode 文件

新生成的 Episode 包含：

```text
episode_000/
├── control_transition.npz
├── control_transition.json
└── preview.mp4
```

- `control_transition.npz`：每个仿真步的目标、当前状态、下一状态和控制力矩。
- `control_transition.json`：关节顺序、采样周期、单位、动作来源和相对文件路径。
- `preview.mp4`：用于人工检查动作效果的 MuJoCo 视频。

`dataset/episodes` 中附带敬礼和挥手的完整 Episode 示例，每个示例都包含 NPZ、JSON 和 MP4。

## 6 校验 Episode

校验刚生成的敬礼数据：

```powershell
python .\export_lerobot_dataset.py `
  --episode .\output\episodes\q1_salute\episode_000 `
  --validate-only
```

校验会检查关节顺序、数组形状、时间连续性、数值有效性以及视频尺寸、帧率和帧数。

## 7 转换为 LeRobot

```powershell
python .\export_lerobot_dataset.py `
  --episode .\output\episodes\q1_salute\episode_000 `
  --output .\output\lerobot\q1_salute `
  --task "Q1 performs a saluting motion"
```

Episode 和 LeRobot 数据集使用不同的输出目录，因此不会发生路径冲突。LeRobot 输出目录同样必须不存在。

## 8 使用关节角度编辑器

```powershell
python .\joint_pose_editor.py
```

编辑器默认加载包内模型，可以拖动滑块或输入精确角度，并复制 22 关节 JSON。如需加载其他模型，可使用：

```powershell
python .\joint_pose_editor.py --model "C:\path\to\model.xml"
```

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from common.joint_map import DEFAULT_MJCF
from episode_generator.motions import DEFAULT_STAND, build_wave, load_csv
from episode_generator.simulation import Q1Simulation

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent


def relative_path(path: str | pathlib.Path) -> str:
    return pathlib.Path(
        os.path.relpath(pathlib.Path(path).resolve(), PACKAGE_ROOT)
    ).as_posix()


class VideoWriter:
    def __init__(self, path: pathlib.Path, width: int, height: int, fps: float):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("找不到 FFmpeg")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, image: np.ndarray) -> None:
        self.process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("FFmpeg 生成视频失败")


def camera_for(model) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 2.25
    camera.azimuth = 140.0
    camera.elevation = -5.0
    camera.lookat[:] = [0.0, 0.0, 0.55]
    return camera


def record(
    simulation: Q1Simulation,
    motion,
    output: pathlib.Path,
    source_csv: str | None,
    csv_fps: float,
    video_fps: float,
    width: int,
    height: int,
) -> None:
    unknown = set(motion.pose_at(0.0)) - set(simulation.joint_names)
    if unknown:
        raise ValueError(f"动作包含未知关节: {sorted(unknown)}")

    home = {name: 0.0 for name in simulation.joint_names} | DEFAULT_STAND
    if source_csv:
        simulation.set_initial_pose(home | motion.pose_at(0.0))

    output.mkdir(parents=True)
    renderer = mujoco.Renderer(simulation.model, height=height, width=width)
    writer = VideoWriter(output / "preview.mp4", width, height, video_fps)
    camera = camera_for(simulation.model)
    dt = simulation.timestep
    steps = round(motion.duration / dt)
    frame_count = int(np.ceil(motion.duration * video_fps - 1e-12))
    render_steps = {round(frame / (video_fps * dt)) for frame in range(frame_count)}
    if len(render_steps) != frame_count:
        raise ValueError("视频帧率不能高于仿真频率")

    data = {
        "timestamp": [],
        "action_position": [],
        "action_velocity": [],
        "action_effort": [],
        "action_stiffness": [],
        "action_damping": [],
        "observation_position": [],
        "observation_velocity": [],
        "next_observation_position": [],
        "next_observation_velocity": [],
        "next_observation_effort": [],
    }
    zeros = np.zeros(len(simulation.joint_names))
    try:
        for step in range(steps + 1):
            time = step * dt
            data["timestamp"].append(time)
            data["observation_position"].append(simulation.positions())
            data["observation_velocity"].append(simulation.velocities())
            simulation.set_target(home | motion.pose_at(time))
            data["action_position"].append(simulation.target.copy())
            data["action_velocity"].append(zeros)
            data["action_effort"].append(zeros)
            data["action_stiffness"].append(simulation.stiffness)
            data["action_damping"].append(simulation.damping)
            simulation.step()
            data["next_observation_position"].append(simulation.positions())
            data["next_observation_velocity"].append(simulation.velocities())
            data["next_observation_effort"].append(simulation.torque.copy())
            if step in render_steps:
                renderer.update_scene(simulation.data, camera)
                writer.write(renderer.render())
    finally:
        renderer.close()
        writer.close()

    arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in data.items()
    }
    if not simulation.is_finite() or not all(
        np.all(np.isfinite(value)) for value in arrays.values()
    ):
        raise RuntimeError("仿真结果含 NaN 或无穷值")
    np.savez_compressed(
        output / "control_transition.npz",
        joint_names=np.asarray(simulation.joint_names, dtype=str),
        **arrays,
    )
    metadata = {
        "motion": motion.name,
        "joint_names": list(simulation.joint_names),
        "num_transitions": steps + 1,
        "timestep_s": dt,
        "alignment": "observation[t] --action[t]--> next_observation[t+1]",
        "model": relative_path(simulation.original_model),
        "mode": "pinned",
        "gravity": True,
        "gravity_compensation": True,
        "position_unit": "rad",
        "velocity_unit": "rad/s",
        "effort_unit": "N*m",
        "stiffness_unit": "N*m/rad",
        "damping_unit": "N*m*s/rad",
        "source_csv": relative_path(source_csv) if source_csv else None,
        "source_csv_fps": csv_fps if source_csv else None,
        "video_fps": video_fps,
        "video_size": [width, height],
        "initialization": "first_csv_frame" if source_csv else "model_qpos0",
    }
    (output / "control_transition.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Q1 MuJoCo 动作 Episode")
    parser.add_argument(
        "--model",
        default=DEFAULT_MJCF,
        help=f"Q1 MJCF 模型（默认 {DEFAULT_MJCF}）",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion", choices=["wave"], help="内置动作")
    source.add_argument("--from-csv", help="动作 CSV")
    parser.add_argument("--csv-fps", type=float, default=120.0)
    parser.add_argument("--output", required=True, help="新 Episode 目录")
    parser.add_argument("--fps", type=float, default=60.0, help="视频帧率")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    output = pathlib.Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在: {output}")
    if args.fps <= 0.0 or args.width <= 0 or args.height <= 0:
        raise ValueError("视频帧率和尺寸必须为正数")
    motion = load_csv(args.from_csv, args.csv_fps) if args.from_csv else build_wave()
    simulation = Q1Simulation(args.model)
    record(
        simulation,
        motion,
        output,
        args.from_csv,
        args.csv_fps,
        args.fps,
        args.width,
        args.height,
    )
    print(f"已生成 {output}")
    print("  control_transition.npz")
    print("  control_transition.json")
    print("  preview.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

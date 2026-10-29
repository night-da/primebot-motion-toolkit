from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets import LeRobotDataset

from common.q1_schema import Q1_JOINT_NAMES

CONTROL_FILENAME = "control_transition.npz"
VIDEO_FILENAME = "preview.mp4"
VECTOR_FIELDS = (
    "action_position",
    "observation_position",
    "observation_velocity",
    "next_observation_position",
    "next_observation_velocity",
)


@dataclass(frozen=True)
class ControlData:
    timestamp: np.ndarray
    action_position: np.ndarray
    observation_position: np.ndarray
    observation_velocity: np.ndarray
    next_observation_position: np.ndarray
    next_observation_velocity: np.ndarray
    timestep: float

    @property
    def transition_count(self) -> int:
        return len(self.timestamp)


@dataclass(frozen=True)
class VideoInfo:
    fps: int
    width: int
    height: int
    frame_count: int


@contextmanager
def open_video(path: Path) -> Iterator[cv2.VideoCapture]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"无法打开视频：{path}")
    try:
        yield capture
    finally:
        capture.release()


def load_control(path: Path) -> ControlData:
    required = ("joint_names", "timestamp", *VECTOR_FIELDS)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"NPZ 缺少字段：{missing}")

        joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
        if joint_names != Q1_JOINT_NAMES:
            raise ValueError("NPZ 的关节名称或顺序与 Q1 的统一 22 关节顺序不一致")

        timestamp = np.asarray(archive["timestamp"], dtype=np.float64)
        vectors = {name: np.asarray(archive[name]) for name in VECTOR_FIELDS}

    if timestamp.ndim != 1 or len(timestamp) < 2:
        raise ValueError("timestamp 必须是一维数组且至少包含两个时间点")
    if not np.all(np.isfinite(timestamp)):
        raise ValueError("timestamp 包含 NaN 或无穷值")

    delta = np.diff(timestamp)
    timestep = float(delta[0])
    if timestep <= 0.0 or not np.allclose(delta, timestep, rtol=0.0, atol=1e-12):
        raise ValueError("timestamp 必须严格递增并采用等间隔采样")

    expected_shape = (len(timestamp), len(Q1_JOINT_NAMES))
    for name, values in vectors.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} 形状为 {values.shape}，预期为 {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} 包含 NaN 或无穷值")

    if not np.array_equal(
        vectors["next_observation_position"][:-1],
        vectors["observation_position"][1:],
    ):
        raise ValueError("相邻 transition 的位置状态不连续")
    if not np.array_equal(
        vectors["next_observation_velocity"][:-1],
        vectors["observation_velocity"][1:],
    ):
        raise ValueError("相邻 transition 的速度状态不连续")

    return ControlData(timestamp=timestamp, timestep=timestep, **vectors)


def inspect_video(path: Path) -> VideoInfo:
    with open_video(path) as capture:
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(reported_fps) or reported_fps <= 0.0:
            raise ValueError("无法读取有效的视频帧率")
        fps = round(reported_fps)
        if abs(reported_fps - fps) > 1e-3:
            raise ValueError(f"视频帧率 {reported_fps:g} 不是整数帧率")

        width = height = frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_height, current_width = frame.shape[:2]
            if frame_count == 0:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise ValueError("视频中存在尺寸不一致的画面")
            frame_count += 1

    if frame_count == 0:
        raise ValueError("视频中没有可读取的画面")
    return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)


def frame_source_indices(control: ControlData, video: VideoInfo) -> np.ndarray:
    duration = float(control.timestamp[-1] - control.timestamp[0])
    expected_frames = round(duration * video.fps)
    if video.frame_count != expected_frames:
        raise ValueError(
            f"视频包含 {video.frame_count} 帧，按控制轨迹时长应为 {expected_frames} 帧"
        )

    rendered = np.rint(
        np.arange(video.frame_count, dtype=np.float64) / (video.fps * control.timestep)
    ).astype(np.int64)
    source = rendered + 1
    if source[-1] >= control.transition_count:
        raise ValueError("视频最后一帧超出 NPZ 控制轨迹范围")
    return source


def dataset_features(video: VideoInfo) -> dict:
    def vector() -> dict:
        return {
            "dtype": "float32",
            "shape": (len(Q1_JOINT_NAMES),),
            "names": list(Q1_JOINT_NAMES),
        }

    return {
        "observation.state": vector(),
        "observation.velocity": vector(),
        "action": vector(),
        "observation.images.sim": {
            "dtype": "video",
            "shape": (3, video.height, video.width),
            "names": ["channels", "height", "width"],
        },
    }


def create_dataset(
    output: Path,
    repo_id: str,
    robot_type: str,
    video: VideoInfo,
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=video.fps,
        robot_type=robot_type,
        features=dataset_features(video),
        use_videos=True,
        image_writer_threads=4,
    )


def remove_empty_images_dir(output: Path) -> None:
    images = output / "images"
    if images.is_dir() and not any(path.is_file() for path in images.rglob("*")):
        shutil.rmtree(images)


def discard_failed_export(dataset: LeRobotDataset | None, output: Path) -> None:
    if dataset is not None and dataset.writer is not None:
        with suppress(Exception):
            dataset.writer.stop_image_writer()
        with suppress(Exception):
            dataset.writer.close_writer()
    with suppress(OSError):
        shutil.rmtree(output)


def add_episode_frames(
    dataset: LeRobotDataset,
    video_path: Path,
    control: ControlData,
    source_indices: np.ndarray,
    task: str,
) -> None:
    with open_video(video_path) as capture:
        for source_index in source_indices:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("视频在导出过程中提前结束")
            dataset.add_frame(
                {
                    "observation.state": control.observation_position[
                        source_index
                    ].astype(np.float32),
                    "observation.velocity": control.observation_velocity[
                        source_index
                    ].astype(np.float32),
                    "action": control.action_position[source_index].astype(np.float32),
                    "observation.images.sim": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    "task": task,
                }
            )


def read_episode(episode: Path) -> tuple[Path, ControlData, VideoInfo, np.ndarray]:
    control_path = episode / CONTROL_FILENAME
    video_path = episode / VIDEO_FILENAME
    for path in (control_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    control = load_control(control_path)
    video = inspect_video(video_path)
    indices = frame_source_indices(control, video)
    return video_path, control, video, indices


def export_dataset(
    episode: Path,
    output: Path,
    task: str,
    repo_id: str,
    robot_type: str,
) -> None:
    if output.exists():
        raise FileExistsError(f"输出目录已经存在：{output}")

    video_path, control, video, indices = read_episode(episode)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = None
    try:
        dataset = create_dataset(output, repo_id, robot_type, video)
        add_episode_frames(dataset, video_path, control, indices, task)
        dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
        remove_empty_images_dir(output)
    except Exception:
        discard_failed_export(dataset, output)
        raise

    print(f"LeRobotDataset v3.0 已生成：{output}")
    print(f"任务：{task}")
    print(f"帧数：{video.frame_count}")
    print(f"视频：{video.width}x{video.height} @ {video.fps} FPS")
    print(f"控制周期：{control.timestep:g} s")


def validate_episode(episode: Path) -> None:
    _, control, video, _ = read_episode(episode)
    print(f"校验通过：{episode}")
    print(f"控制记录：{control.transition_count} 条 @ {1.0 / control.timestep:g} Hz")
    print(
        f"视频：{video.frame_count} 帧，{video.width}x{video.height} @ {video.fps} FPS"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Q1 控制 Episode 转换为 LeRobotDataset"
    )
    parser.add_argument("--episode", type=Path, required=True, help="原始 Episode 目录")
    parser.add_argument("--output", type=Path, help="LeRobotDataset 输出目录")
    parser.add_argument("--task", help="动作的自然语言任务描述")
    parser.add_argument("--repo-id", help="数据集标识，默认使用 local/<输出目录名>")
    parser.add_argument("--robot-type", default="primebot_q1", help="机器人类型")
    parser.add_argument(
        "--validate-only", action="store_true", help="只校验，不生成数据集"
    )
    args = parser.parse_args()

    if not args.validate_only:
        if args.output is None:
            parser.error("生成数据时必须提供 --output")
        if not args.task or not args.task.strip():
            parser.error("生成数据时必须提供非空的 --task")
        args.task = args.task.strip()
    return args


def main() -> None:
    args = parse_args()
    episode = args.episode.resolve()
    if args.validate_only:
        validate_episode(episode)
        return

    output = args.output.resolve()
    export_dataset(
        episode=episode,
        output=output,
        task=args.task,
        repo_id=args.repo_id or f"local/{output.name}",
        robot_type=args.robot_type,
    )


if __name__ == "__main__":
    main()

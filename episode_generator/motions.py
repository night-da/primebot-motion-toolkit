from __future__ import annotations

import csv
import itertools
import math
import pathlib

from .trajectory import Hold, Motion, Move, Sine, Track

UPPER_JOINTS = [
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
DEFAULT_STAND = dict(
    zip(
        UPPER_JOINTS,
        [
            -0.01982023485863345,
            -0.00021580613966982196,
            0.017233758101518008,
            0.22511493021261703,
            -0.003530979547924497,
            1.0186741542682025,
            0.017164215358473386,
            -0.22262401216020816,
            0.004204617106736518,
            1.0185279282650144,
        ],
    )
)


def build_wave() -> Motion:
    stand = dict(DEFAULT_STAND)
    arm = {
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_yaw_joint": math.radians(-85.0),
        "right_elbow_joint": math.radians(35.0),
    }
    center = math.radians(-138.0)
    raised = stand | arm | {"right_shoulder_roll_joint": center}
    return Motion(
        "wave",
        [
            Move({}, stand, 2.0),
            Move(stand, raised, 2.0),
            Sine(
                stand,
                {"right_shoulder_roll_joint": (center, math.radians(18.0))},
                arm,
                1.5,
                4.0,
            ),
            Move(raised, stand, 2.0),
            Hold(stand, 1.0),
        ],
    )


def load_csv(path: str, fps: float) -> Motion:
    source = pathlib.Path(path)
    with source.open(newline="", encoding="utf-8") as file:
        rows = csv.reader(file)
        header = [cell.strip() for cell in next(rows)]
        if (
            not header
            or len(set(header)) != len(header)
            or any(not name for name in header)
        ):
            raise ValueError("CSV 表头为空、重复或包含空列名")
        has_time = header[0].lower() in {"t", "time", "timestamp", "timestamp_s"}
        if not has_time and (not math.isfinite(fps) or fps <= 0.0):
            raise ValueError("无时间列的 CSV 必须提供正数帧率")
        joints = header[1:] if has_time else header
        times: list[float] = []
        poses: list[dict] = []
        for line, row in enumerate(rows, 2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV 第 {line} 行列数不正确")
            values = [float(cell) for cell in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"CSV 第 {line} 行含非有限数值")
            times.append(values[0] if has_time else len(poses) / fps)
            poses.append(dict(zip(joints, values[1:] if has_time else values)))
    if not poses:
        raise ValueError("CSV 没有数据行")
    if any(end <= start for start, end in itertools.pairwise(times)):
        raise ValueError("CSV 时间必须严格递增")
    if not has_time:
        times.append(len(poses) / fps)
        poses.append(dict(poses[-1]))
    return Motion(source.stem, [Track(times, poses)])

from __future__ import annotations

import bisect
import math


def interpolate(start: dict, end: dict, ratio: float) -> dict:
    keys = set(start) | set(end)
    return {
        name: start.get(name, 0.0) + (end.get(name, 0.0) - start.get(name, 0.0)) * ratio
        for name in keys
    }


class Hold:
    def __init__(self, pose: dict, duration: float):
        self.pose = dict(pose)
        self.duration = float(duration)

    def pose_at(self, _time: float) -> dict:
        return dict(self.pose)


class Move:
    def __init__(self, start: dict, end: dict, duration: float):
        self.start = dict(start)
        self.end = dict(end)
        self.duration = float(duration)

    def pose_at(self, time: float) -> dict:
        ratio = min(1.0, max(0.0, time / self.duration))
        ratio = 0.5 - 0.5 * math.cos(math.pi * ratio)
        return interpolate(self.start, self.end, ratio)


class Sine:
    def __init__(
        self,
        base: dict,
        oscillate: dict,
        fixed: dict,
        cycles: float,
        duration: float,
    ):
        self.base = dict(base)
        self.oscillate = dict(oscillate)
        self.fixed = dict(fixed)
        self.cycles = float(cycles)
        self.duration = float(duration)

    def pose_at(self, time: float) -> dict:
        ratio = min(1.0, max(0.0, time / self.duration))
        wave = math.sin(2.0 * math.pi * self.cycles * ratio)
        pose = self.base | self.fixed
        for name, (center, amplitude) in self.oscillate.items():
            pose[name] = center + amplitude * wave
        return pose


class Track:
    def __init__(self, times: list[float], poses: list[dict]):
        self.times = times
        self.poses = poses
        self.duration = times[-1] - times[0]

    def pose_at(self, time: float) -> dict:
        if len(self.times) == 1:
            return dict(self.poses[0])
        time = self.times[0] + min(max(time, 0.0), self.duration)
        index = min(bisect.bisect_right(self.times, time) - 1, len(self.times) - 2)
        start, end = self.times[index : index + 2]
        return interpolate(
            self.poses[index],
            self.poses[index + 1],
            (time - start) / (end - start),
        )


class Motion:
    def __init__(self, name: str, segments: list):
        if not segments or any(segment.duration <= 0.0 for segment in segments):
            raise ValueError("动作必须包含时长为正数的片段")
        self.name = name
        self.segments = segments
        self.duration = sum(segment.duration for segment in segments)

    def pose_at(self, time: float) -> dict:
        elapsed = 0.0
        for segment in self.segments:
            if time < elapsed + segment.duration:
                return segment.pose_at(time - elapsed)
            elapsed += segment.duration
        return self.segments[-1].pose_at(self.segments[-1].duration)

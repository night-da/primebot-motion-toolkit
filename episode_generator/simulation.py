from __future__ import annotations

import hashlib
import pathlib
import re
import tempfile

import mujoco
import numpy as np

from common.joint_map import load_model_and_mapping

GAINS = {
    "leg": (220.0, 12.0),
    "ankle": (220.0, 3.0),
    "waist": (150.0, 8.0),
    "arm": (60.0, 3.0),
}


def gain_for(name: str) -> tuple[float, float]:
    if "ankle" in name:
        return GAINS["ankle"]
    if "hip" in name or "knee" in name:
        return GAINS["leg"]
    if "waist" in name:
        return GAINS["waist"]
    return GAINS["arm"]


def pinned_model(source: pathlib.Path) -> pathlib.Path:
    text = source.read_text(encoding="utf-8")
    if "</mujoco>" not in text:
        raise ValueError("模型缺少 </mujoco>")
    mesh_dir = (source.parent / "../meshes").resolve().as_posix()
    text = re.sub('meshdir="[^"]*"', f'meshdir="{mesh_dir}/"', text, count=1)
    weld = '  <equality>\n    <weld body1="pelvis" name="pin_base"/>\n  </equality>\n'
    text = text.replace("</mujoco>", weld + "</mujoco>", 1)
    key = hashlib.sha1(
        f"{source.resolve()}|{source.stat().st_mtime_ns}".encode()
    ).hexdigest()[:12]
    output = pathlib.Path(tempfile.gettempdir()) / f"q1_pinned_{key}.xml"
    if not output.exists():
        output.write_text(text, encoding="utf-8")
    return output


class Q1Simulation:
    def __init__(self, model_path: str):
        self.original_model = pathlib.Path(model_path).resolve()
        self.model, self.joint_names, self.mapping = load_model_and_mapping(
            pinned_model(self.original_model)
        )
        self.data = mujoco.MjData(self.model)
        self.stiffness = np.array([gain_for(name)[0] for name in self.joint_names])
        self.damping = np.array([gain_for(name)[1] for name in self.joint_names])
        self.target = self.positions()
        self.torque = np.zeros(len(self.joint_names))

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def positions(self) -> np.ndarray:
        return np.array(
            [
                self.data.qpos[self.mapping[name]["qpos_adr"]]
                for name in self.joint_names
            ]
        )

    def velocities(self) -> np.ndarray:
        return np.array(
            [self.data.qvel[self.mapping[name]["dof_adr"]] for name in self.joint_names]
        )

    def set_initial_pose(self, pose: dict[str, float]) -> None:
        self.set_target(pose)
        for name, value in zip(self.joint_names, self.target):
            self.data.qpos[self.mapping[name]["qpos_adr"]] = value
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, pose: dict[str, float]) -> None:
        unknown = set(pose) - set(self.joint_names)
        if unknown:
            raise ValueError(f"模型中不存在这些关节: {sorted(unknown)}")
        target = self.target.copy()
        for name, value in pose.items():
            entry = self.mapping[name]
            if entry["range_rad"] is not None:
                low, high = entry["range_rad"]
                if not low <= value <= high:
                    raise ValueError(f"{name}={value:.6f} 超出模型范围 [{low}, {high}]")
            target[self.joint_names.index(name)] = value
        self.target = target

    def step(self) -> None:
        position = self.positions()
        velocity = self.velocities()
        self.torque = (
            self.stiffness * (self.target - position) - self.damping * velocity
        )
        for index, name in enumerate(self.joint_names):
            entry = self.mapping[name]
            self.torque[index] += self.data.qfrc_bias[entry["dof_adr"]]
            low, high = entry["ctrlrange"]
            self.torque[index] = np.clip(self.torque[index], low, high)
            self.data.ctrl[entry["actuator_id"]] = self.torque[index]
        mujoco.mj_step(self.model, self.data)

    def is_finite(self) -> bool:
        return bool(
            np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
        )

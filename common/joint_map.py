from __future__ import annotations

from pathlib import Path

import mujoco

from .q1_schema import Q1_JOINT_NAMES

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = PACKAGE_ROOT / "model" / "xml" / "q1_v3_mc.xml"


def mjcf_joint_names(model: mujoco.MjModel) -> list[str]:
    joint_names = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        if joint_name is None:
            raise ValueError(f"MJCF 中的关节 {joint_id} 没有名称")
        joint_names.append(joint_name)

    if tuple(joint_names) != Q1_JOINT_NAMES:
        raise ValueError(
            f"MJCF 的关节名称或顺序与 Q1 的统一 22 关节顺序不一致：{joint_names}"
        )
    return joint_names


def build_mapping(model: mujoco.MjModel, joint_names: list[str]) -> dict:
    actuator_by_joint = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        if joint_id >= 0 and joint_id not in actuator_by_joint:
            actuator_by_joint[joint_id] = actuator_id

    mapping = {}
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise ValueError(f"MJCF 中找不到关节：{joint_name}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Q1 关节不是旋转关节：{joint_name}")
        if not model.jnt_limited[joint_id]:
            raise ValueError(f"Q1 关节没有角度范围：{joint_name}")

        actuator_id = actuator_by_joint.get(joint_id)
        if actuator_id is None:
            raise ValueError(f"Q1 关节没有执行器：{joint_name}")

        lower_rad, upper_rad = model.jnt_range[joint_id]
        lower_ctrl, upper_ctrl = model.actuator_ctrlrange[actuator_id]
        mapping[joint_name] = {
            "joint_id": joint_id,
            "qpos_adr": int(model.jnt_qposadr[joint_id]),
            "dof_adr": int(model.jnt_dofadr[joint_id]),
            "actuator_id": actuator_id,
            "range_rad": (float(lower_rad), float(upper_rad)),
            "ctrlrange": (float(lower_ctrl), float(upper_ctrl)),
        }
    return mapping


def load_model_and_mapping(mjcf_path: str | Path):
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    joint_names = mjcf_joint_names(model)
    return model, joint_names, build_mapping(model, joint_names)

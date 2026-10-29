from __future__ import annotations

import argparse
import json
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np

from common.joint_map import DEFAULT_MJCF, build_mapping, mjcf_joint_names

ANGLE_DECIMALS = 4
LIMIT_DECIMALS = 2
VIEWER_SYNC_MS = 16
WINDOW_SIZE = "760x860"
WINDOW_MIN_SIZE = (620, 480)


def model_limit_deg(value_rad: float) -> float:
    return round(float(np.rad2deg(value_rad)), LIMIT_DECIMALS)


def load_editor_model(model_path: Path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_names = mjcf_joint_names(model)
    mapping = build_mapping(model, joint_names)
    return model, joint_names, mapping


@dataclass
class JointField:
    name: str
    qpos_adr: int
    lower_deg: float
    upper_deg: float
    angle_var: tk.DoubleVar
    text_var: tk.StringVar


class Q1JointPoseEditor:
    def __init__(self, model_path: Path):
        self.model, self.joint_names, self.mapping = load_editor_model(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.root = tk.Tk()
        self.root.title("Q1 关节角度调节器")
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*WINDOW_MIN_SIZE)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.fields: dict[str, JointField] = {}
        self.status_text = tk.StringVar(
            value="拖动滑块，或在输入框填写精确角度后按 Enter。"
        )
        self.default_font = tkfont.nametofont("TkDefaultFont")
        self.text_font = tkfont.nametofont("TkTextFont")
        self.fixed_font = tkfont.nametofont("TkFixedFont")
        self.input_font = tkfont.Font(family="Consolas", size=10)
        self.title_font = tkfont.Font(
            family="Microsoft YaHei UI", size=15, weight="bold"
        )
        self._last_font_size: int | None = None

        self._build_ui()
        self.root.bind("<Configure>", self._on_window_resize)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._configure_camera()
        self._update_pose()
        self.root.after(VIEWER_SYNC_MS, self._sync_viewer)

    def _build_ui(self) -> None:
        self._build_header()
        self._build_joint_panel()

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill="x")

        ttk.Label(
            header,
            text="Q1 关节角度调节器",
            font=self.title_font,
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="所有滑块单位均为度（°）；模型保持静态，不执行重力仿真。",
        ).pack(anchor="w", pady=(3, 8))

        buttons = ttk.Frame(header)
        buttons.pack(fill="x")
        ttk.Button(
            buttons,
            text="恢复模型默认姿态",
            command=self._reset_pose,
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="应用所有输入",
            command=self._apply_all_entries,
        ).pack(side="left", padx=6)
        ttk.Button(
            buttons,
            text="复制 22 关节 JSON",
            command=self._copy_json,
        ).pack(side="left", padx=6)

        ttk.Label(header, textvariable=self.status_text).pack(anchor="w", pady=(10, 0))

    def _build_joint_panel(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        slider_frame = ttk.Frame(canvas)
        slider_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window_id = canvas.create_window((0, 0), window=slider_frame, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for row, joint_name in enumerate(self.joint_names):
            self._add_joint_row(slider_frame, row, joint_name)

    def _add_joint_row(
        self,
        parent: ttk.Frame,
        row: int,
        joint_name: str,
    ) -> None:
        joint_info = self.mapping[joint_name]
        lower_rad, upper_rad = joint_info["range_rad"]
        lower_deg = model_limit_deg(lower_rad)
        upper_deg = model_limit_deg(upper_rad)
        angle_deg = round(
            float(np.rad2deg(self.data.qpos[joint_info["qpos_adr"]])),
            ANGLE_DECIMALS,
        )

        field = JointField(
            name=joint_name,
            qpos_adr=joint_info["qpos_adr"],
            lower_deg=lower_deg,
            upper_deg=upper_deg,
            angle_var=tk.DoubleVar(value=angle_deg),
            text_var=tk.StringVar(value=self._format_angle(angle_deg)),
        )
        self.fields[joint_name] = field

        line = ttk.Frame(parent, padding=(0, 4))
        line.grid(row=row, column=0, sticky="ew")
        line.columnconfigure(1, weight=1)

        ttk.Label(line, text=joint_name, width=29).grid(row=0, column=0, sticky="w")
        ttk.Scale(
            line,
            from_=lower_deg,
            to=upper_deg,
            variable=field.angle_var,
            command=lambda _value, name=joint_name: self._on_slider(name),
        ).grid(row=0, column=1, sticky="ew", padx=6)

        input_box = tk.Entry(
            line,
            textvariable=field.text_var,
            width=12,
            justify="center",
            font=self.input_font,
            relief="solid",
            borderwidth=1,
        )
        input_box.grid(row=0, column=2, sticky="e", padx=(0, 3))
        input_box.bind(
            "<Return>",
            lambda _event, name=joint_name: self._apply_entry(name),
        )
        ttk.Label(
            line,
            text=f"°  [{lower_deg:g}, {upper_deg:g}]",
            width=17,
            anchor="w",
        ).grid(row=0, column=3, sticky="w")

    @staticmethod
    def _format_angle(value: float) -> str:
        return f"{value:.{ANGLE_DECIMALS}f}"

    def _sync_angle_text(self, field: JointField) -> None:
        field.text_var.set(self._format_angle(field.angle_var.get()))

    def _parse_angle(self, field: JointField) -> tuple[float | None, str | None]:
        text = field.text_var.get().strip()
        try:
            value = float(text)
        except ValueError:
            return None, f"{field.name}：请输入有效数字。"

        if not np.isfinite(value):
            return None, f"{field.name}：请输入有限数值。"
        if value < field.lower_deg or value > field.upper_deg:
            return (
                None,
                (
                    f"{field.name}：{value:g}° 超出范围 "
                    f"[{field.lower_deg:g}, {field.upper_deg:g}]°。"
                ),
            )
        return round(value, ANGLE_DECIMALS), None

    def _on_slider(self, joint_name: str) -> None:
        field = self.fields[joint_name]
        field.angle_var.set(round(field.angle_var.get(), ANGLE_DECIMALS))
        self._sync_angle_text(field)
        self._update_pose()

    def _apply_entry(self, joint_name: str) -> None:
        field = self.fields[joint_name]
        value, error = self._parse_angle(field)
        if error is not None:
            self._sync_angle_text(field)
            self.status_text.set(f"{error} 未应用。")
            return

        assert value is not None
        self._set_field_angle(field, value)
        self._update_pose()
        self.status_text.set(f"已应用 {field.name} = {self._format_angle(value)}°。")

    def _apply_all_entries(self) -> None:
        values: dict[str, float] = {}
        for field in self.fields.values():
            value, error = self._parse_angle(field)
            if error is not None:
                self.status_text.set(f"{error} 所有输入均未应用。")
                return
            assert value is not None
            values[field.name] = value

        self._set_degrees(values)
        self.status_text.set("已应用全部 22 个输入框中的角度。")

    def _set_field_angle(self, field: JointField, value: float) -> None:
        field.angle_var.set(round(value, ANGLE_DECIMALS))
        self._sync_angle_text(field)

    def _set_degrees(self, values: dict[str, float]) -> None:
        for joint_name, value in values.items():
            field = self.fields.get(joint_name)
            if field is not None:
                self._set_field_angle(field, value)
        self._update_pose()

    def _write_pose_to_model(self) -> None:
        for field in self.fields.values():
            self.data.qpos[field.qpos_adr] = np.deg2rad(field.angle_var.get())
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _update_pose(self) -> None:
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            self._write_pose_to_model()
            return
        with viewer.lock():
            self._write_pose_to_model()

    def _reset_pose(self) -> None:
        values = {
            field.name: float(np.rad2deg(self.model.qpos0[field.qpos_adr]))
            for field in self.fields.values()
        }
        self._set_degrees(values)
        self.status_text.set("已恢复模型默认姿态。")

    def _copy_json(self) -> None:
        values = {
            field.name: {
                "degree": round(field.angle_var.get(), ANGLE_DECIMALS),
                "radian": round(float(np.deg2rad(field.angle_var.get())), 9),
            }
            for field in self.fields.values()
        }
        self.root.clipboard_clear()
        self.root.clipboard_append(json.dumps(values, ensure_ascii=False, indent=2))
        self.status_text.set("已复制 22 关节 JSON（同时包含度和弧度）。")

    def _configure_camera(self) -> None:
        with self.viewer.lock():
            self.viewer.cam.distance = 1.65
            self.viewer.cam.azimuth = 150.0
            self.viewer.cam.elevation = -5.0
            self.viewer.cam.lookat[:] = [0.0, -0.03, 0.72]

    def _on_window_resize(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        scale = min(event.width / 760.0, event.height / 860.0)
        font_size = max(8, min(18, round(10 * scale)))
        if font_size == self._last_font_size:
            return

        self._last_font_size = font_size
        self.default_font.configure(size=font_size)
        self.text_font.configure(size=font_size)
        self.fixed_font.configure(size=font_size)
        self.input_font.configure(size=font_size)
        self.title_font.configure(size=max(12, min(26, round(font_size * 1.5))))

    def _sync_viewer(self) -> None:
        if not self.viewer.is_running():
            self.close()
            return
        self.viewer.sync()
        self.root.after(VIEWER_SYNC_MS, self._sync_viewer)

    def close(self) -> None:
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            viewer.close()
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="用角度滑块静态编辑 Q1 的 22 关节姿态")
    parser.add_argument("--model", type=Path, default=DEFAULT_MJCF)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查模型和关节映射，不打开窗口",
    )
    args = parser.parse_args()

    if args.check:
        model, joint_names, mapping = load_editor_model(args.model)
        print(f"模型: {args.model.resolve()}")
        print(f"可调关节: {len(joint_names)}")
        for joint_name in joint_names:
            lower_rad, upper_rad = mapping[joint_name]["range_rad"]
            lower_deg = model_limit_deg(lower_rad)
            upper_deg = model_limit_deg(upper_rad)
            print(
                f"  {joint_name}: "
                f"{lower_deg:.{ANGLE_DECIMALS}f} .. "
                f"{upper_deg:.{ANGLE_DECIMALS}f} deg"
            )
        print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
        return

    Q1JointPoseEditor(args.model).run()


if __name__ == "__main__":
    main()

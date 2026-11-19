"""Generate 50 Hz standing reaches using IK and forward physics."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from supermimic.robots.go2_piper import (
    ARM_KD, ARM_KP, ASSETS, BASE, EFFORT, HOME, JOINTS, LEGS, LEG_KD, LEG_KP, MODEL, SIM, TIP,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FPS = round(1 / SIM.control_dt)
DWELL_START, DWELL_END = 1.45, 1.70
OFFSETS = {
    "train": ((.07, 0, 0), (.05, .04, 0), (.05, -.04, 0), (.055, 0, .04),
              (.055, 0, -.035), (.035, .025, .025)),
    "holdout": ((.06, .025, .02), (.045, -.03, -.02)),
}


def prepare_model() -> None:
    root = ET.parse(ASSETS / "go2piper.xml").getroot()
    compiler = root.find("compiler")
    compiler.attrib.pop("meshdir", None)
    for tag in ("actuator", "sensor", "keyframe", "option"):
        node = root.find(tag)
        if node is not None:
            root.remove(node)
    for geom in root.iter("geom"):
        # Warp's multi-contact collision detection requires zero geom margins.
        if "margin" in geom.attrib:
            geom.set("margin", "0")
    base = root.find(f"./worldbody/body[@name='{BASE}']")
    base.set("pos", "0 0 0.32")
    base.attrib.pop("euler", None)
    ET.ElementTree(root).write(MODEL, encoding="utf-8")


def floor_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(MODEL))
    spec.option.timestep = SIM.timestep
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, .05),
                            friction=(.8, .02, .01), rgba=(.55, .58, .62, 1))
    return spec.compile()


def arm_ik(model, stance, destination):
    ik = mujoco.MjData(model)
    ik.qpos[:] = stance
    low, high = model.jnt_range[-6:].T
    jac = np.empty((3, model.nv))
    for _ in range(100):
        mujoco.mj_forward(model, ik)
        error = destination - ik.body(TIP).xpos
        if np.linalg.norm(error) < 0.001:
            return ik.qpos[-6:].copy()
        mujoco.mj_jacBody(model, ik, jac, None, model.body(TIP).id)
        arm = jac[:, -6:]
        step = arm.T @ np.linalg.solve(arm @ arm.T + .001 * np.eye(3), error)
        ik.qpos[-6:] = np.clip(ik.qpos[-6:] + np.clip(step, -.06, .06), low + .02, high - .02)
    raise ValueError(f"IK failed to reach {destination.tolist()}")


def interpolate(time: float) -> float:
    phase = np.clip(min((time - .25) / .85, (2.60 - time) / .85), 0, 1)
    return float(phase**3 * (10 - 15 * phase + 6 * phase**2))


def record(model, offset, path: Path, *, calibrate=False) -> dict:
    data = mujoco.MjData(model)
    data.qpos[7:] = list(HOME.values())
    feet = [model.geom(leg).id for leg in LEGS]
    mujoco.mj_forward(model, data)
    data.qpos[2] += .025 - data.geom_xpos[feet, 2].min()
    kp = np.array([LEG_KP] * 12 + [ARM_KP] * 6)
    kd = np.array([LEG_KD] * 12 + [ARM_KD] * 6)
    effort = np.array(EFFORT)
    home = np.array(list(HOME.values()))

    def apply_target(target):
        data.qfrc_applied[6:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
        mujoco.mj_step(model, data)

    for _ in range(160):
        apply_target(home)
    mujoco.mj_forward(model, data)
    start_tip = data.body(TIP).xpos.copy()
    goal = start_tip + offset
    stance = data.qpos.copy()
    goal_angles = arm_ik(model, stance, goal)
    amounts = [interpolate(index / FPS) for index in range(150)]

    def joint_target(amount, angles):
        target = home.copy()
        target[-6:] += amount * (angles - home[-6:])
        return target

    calibration_errors = []
    if calibrate:
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, state, state_spec)
        ik_target = goal.copy()
        for attempt in range(8):
            mujoco.mj_setState(model, data, state, state_spec)
            held = []
            for index, amount in enumerate(amounts[:round(DWELL_END * FPS) + 1]):
                target = joint_target(amount, goal_angles)
                for _ in range(SIM.decimation):
                    apply_target(target)
                if index / FPS >= DWELL_START:
                    mujoco.mj_forward(model, data)
                    held.append(data.body(TIP).xpos.copy())
            residual = goal - np.mean(held, axis=0)
            calibration_errors.append(float(np.linalg.norm(residual)))
            if calibration_errors[-1] <= .0015:
                break
            if attempt < 7:
                ik_target += residual
                goal_angles = arm_ik(model, stance, ik_target)
        mujoco.mj_setState(model, data, state, state_spec)
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    fields = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    out = {field: [] for field in fields}
    tip_error, tilt, contacts, teacher_targets = [], [], [], []
    for index, amount in enumerate(amounts):
        time = index / FPS
        target = joint_target(amount, goal_angles)
        teacher_targets.append(target)
        for _ in range(SIM.decimation):
            apply_target(target)
        mujoco.mj_forward(model, data)
        velocity = np.zeros((model.nbody - 1, 6))
        for body_id, vector in enumerate(velocity, 1):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vector, 0)
        for field, values in zip(fields, (data.qpos[7:], data.qvel[6:], data.xpos[1:], data.xquat[1:],
                                           velocity[:, 3:], velocity[:, :3])):
            out[field].append(values.copy())
        rotation = data.body(BASE).xmat.reshape(3, 3)
        tilt.append(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
        contacts.append(len({geom_id for contact in data.contact for geom_id in contact.geom if geom_id in feet}))
        if DWELL_START <= time <= DWELL_END:
            tip_error.append(np.linalg.norm(data.body(TIP).xpos - goal))

    q = np.asarray(out["joint_pos"])
    joint_ok = ((q >= model.jnt_range[1:, 0] - .02) & (q <= model.jnt_range[1:, 1] + .02)).all()
    metrics = {
        "clip": f"{path.parent.name}/{path.stem}", "offset_m": list(offset), "reach_error_m": float(np.mean(tip_error)),
        "max_tilt_deg": float(max(tilt)), "full_support_fraction": float(np.mean(np.array(contacts) == 4)),
        "joints_within_limits": bool(joint_ok),
    }
    if calibrate:
        metrics["calibration_residual_m"] = calibration_errors
    if not joint_ok or metrics["reach_error_m"] > .055 or max(tilt) > 12 or metrics["full_support_fraction"] < .75:
        raise ValueError(f"Reference failed validation: {metrics}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: np.asarray(values, dtype=np.float32) for key, values in out.items()},
        joint_names=np.array(JOINTS), body_names=np.array(body_names), fps=FPS, reach_goal_w=goal,
        teacher_joint_target=np.asarray(teacher_targets, dtype=np.float32),
        foot_contact_count=np.asarray(contacts, dtype=np.uint8),
    )
    return metrics


def main() -> None:
    prepare_model()
    model = floor_model()
    assert [model.joint(i).name for i in range(1, model.njnt)] == list(JOINTS)
    assert model.nq == 25 and model.nv == 24
    results = [
        record(model, offset, DATA / group / f"reach_{index:02d}.npz")
        for group, offsets in OFFSETS.items() for index, offset in enumerate(offsets)
    ]
    summary = json.dumps(results, indent=2)
    (ROOT / "artifacts" / "reference_metrics.json").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()

"""Simulation-only 18-joint model for the standing reach task."""

from pathlib import Path

from supermimic.robots.spec import (
    DomainRandomization, ObsNoise, ResetNoise, RobotSpec, SimSettings, StateScales,
)

ASSETS = Path(__file__).resolve().parents[4] / "go2_piper"
MODEL = ASSETS / "training_model.xml"
LEGS = ("FL", "FR", "RL", "RR")
LEG_JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 7))
JOINTS = LEG_JOINTS + ARM_JOINTS
BASE = "base_link"
TIP = "end_effector"
HOME = dict(zip(JOINTS, (0.0, 0.8, -1.55) * 4 + (0.0, 0.6, -1.4, 0.0, 0.45, 0.0)))
STAND_HEIGHT = 0.32
LEG_KP, LEG_KD = 65.0, 2.5
ARM_KP, ARM_KD = 45.0, 1.0
EFFORT = (23.7, 23.7, 45.43) * 4 + (20., 20., 15., 7., 5., 5.)
SIM = SimSettings(timestep=0.005, decimation=4, nconmax=36, njmax=180, iterations=10)


def _model_spec():
    import mujoco

    return mujoco.MjSpec.from_file(str(MODEL))


def _entity_config(domain_randomization: bool = False):
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    limits = tuple(zip(JOINTS, EFFORT))
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            stiffness=LEG_KP if name in LEG_JOINTS else ARM_KP,
            damping=LEG_KD if name in LEG_JOINTS else ARM_KD,
            effort_limit=limit,
        )
        for name, limit in limits
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, STAND_HEIGHT), joint_pos=HOME, joint_vel={".*": 0.0},
        ),
        spec_fn=_model_spec,
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.98),
    )


def go2_piper_spec() -> RobotSpec:
    return RobotSpec(
        name="go2_piper", xml_path=MODEL, joint_names=JOINTS,
        root_body=BASE, anchor_body=BASE, com_body=BASE,
        tracked_bodies=(
            BASE, *(f"{leg}_foot" for leg in LEGS), "link2", "link3", "link5", TIP,
        ),
        foot_geom_regex="^(FL|FR|RL|RR)$",
        action_scale=0.35, entity_cfg_fn=_entity_config,
        root_height_threshold=0.11, root_tilt_threshold=0.16,
        # Starting scales for slow, four-foot-supported reaches.
        state_scales=StateScales(
            err_scale=(0.02, 0.05, 0.1, 1.0, 0.1, 3.0),
            out_scale=(0.1, 1.0, 3.0, 0.001, 0.005, 0.02),
            in_scale=(0.32, 1.0, 0.5, 1.0, 1.0, 2.5),
            xi_friction=(0.8, 0.4), xi_bias=(0.0, 0.01), xi_com=(0.0, 0.03),
        ),
        obs_noise=ObsNoise(0.005, 0.1, 0.05, 0.01),
        dr=DomainRandomization((0.6, 1.0), 0.005, ((0.0, 0.0),) * 3, eval_friction=0.8),
        reset_noise=ResetNoise({}, {}, (0.0, 0.0)),
        reward_std_body_pos=0.08, reward_std_body_ori=0.4,
        reward_std_body_lin_vel=0.6, reward_std_body_ang_vel=3.14,
        reward_std_root_pos=0.10, reward_std_root_ori=0.25,
        reward_std_joint_pos=0.25, reward_w_joint_pos=1.0,
        sim=SIM, stand_height=STAND_HEIGHT,
    )

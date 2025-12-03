import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class DofbotReachEnv(gym.Env):
    """
    Dofbot reach environment (PyBullet + Gymnasium)

    Control:
    - Delta-angle POSITION_CONTROL on all revolute joints

    Observation:
    - q, qd, ee_pos, target_pos   (2n + 6)

    Reward:
    - distance improvement
    - pose regularization
    - small control cost
    - orientation encouragement
    """

    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(
        self,
        render_mode=None,
        time_step: float = 1.0 / 240.0,
        n_substeps: int = 4,
        max_episode_steps: int = 300,
        distance_threshold: float = 0.03,
        torque_limit: float = 5.0,  # kept for backward compatibility
    ):
        super().__init__()

        self.render_mode = render_mode
        self.time_step = time_step
        self.n_substeps = n_substeps
        self.max_episode_steps = max_episode_steps
        self.distance_threshold = distance_threshold

        # ---------- PyBullet ----------
        if render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)

        p.setPhysicsEngineParameter(
            numSolverIterations=200,
            fixedTimeStep=self.time_step,
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # ---------- Load DOFBOT ----------
        base_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(
            base_dir, "dofbot", "assets", "dofbot", "dofbot.urdf"
        )

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        # ---------- Joints ----------
        self.joint_indices = []
        lo_all, hi_all = [], []

        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(j)
                lo, hi = info[8], info[9]
                if lo > hi:
                    lo, hi = -math.pi, math.pi
                lo_all.append(lo)
                hi_all.append(hi)

        self.joint_indices = np.array(self.joint_indices, dtype=int)
        self.joint_lower = np.array(lo_all, dtype=np.float32)
        self.joint_upper = np.array(hi_all, dtype=np.float32)
        self.n_joints = len(self.joint_indices)

        # EE = last revolute joint
        self.ee_link = int(self.joint_indices[-1])

        # ---------- Reference pose ----------
        q_ref = np.array([0.0, 0.4, -0.8, 0.6, 0.0], dtype=np.float32)
        if len(q_ref) < self.n_joints:
            q_ref = np.concatenate(
                [q_ref, np.zeros(self.n_joints - len(q_ref))]
            )
        self.q_ref = q_ref[: self.n_joints]

        # Joint force limits
        self.max_forces = np.array(
            [25, 25, 20, 10, 5], dtype=np.float32
        )
        if len(self.max_forces) < self.n_joints:
            self.max_forces = np.pad(
                self.max_forces,
                (0, self.n_joints - len(self.max_forces)),
                constant_values=10.0,
            )
        self.max_forces = self.max_forces[: self.n_joints]

        # Delta angle scale
        self.delta_scale = np.array(
            [0.06, 0.06, 0.06, 0.04, 0.04], dtype=np.float32
        )
        if len(self.delta_scale) < self.n_joints:
            self.delta_scale = np.pad(
                self.delta_scale,
                (0, self.n_joints - len(self.delta_scale)),
                constant_values=0.04,
            )
        self.delta_scale = self.delta_scale[: self.n_joints]

        # Disable default motors
        for j in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id, int(j), p.VELOCITY_CONTROL, force=0
            )

        # ---------- Target ----------
        self.target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.015,
            rgbaColor=[1, 0, 0, 1],
        )
        self.target_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.target_visual,
            basePosition=[0.3, 0, 0.15],
        )

        # ---------- Spaces ----------
        obs_dim = 2 * self.n_joints + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        self.step_count = 0
        self.prev_dist = None
        self.target_pos = None

    # ======================================================
    # Helper functions
    # ======================================================
    def _get_joint_state(self):
        q, qd = [], []
        for j in self.joint_indices:
            s = p.getJointState(self.robot_id, int(j))
            q.append(s[0])
            qd.append(s[1])
        return np.array(q, np.float32), np.array(qd, np.float32)

    def _get_ee_pos_and_rot(self):
        state = p.getLinkState(
            self.robot_id, self.ee_link, computeForwardKinematics=True
        )
        pos = np.array(state[0], dtype=np.float32)
        quat = state[1]
        rot = np.array(
            p.getMatrixFromQuaternion(quat),
            dtype=np.float32,
        ).reshape(3, 3)
        return pos, rot

    # ======================================================
    # ✅ NEW: unbiased target sampling (360°, annulus)
    # ======================================================
    def _sample_target(self):
        r_min = 0.20
        r_max = 0.32
        z = np.random.uniform(0.10, 0.20)

        theta = np.random.uniform(-np.pi, np.pi)
        r = np.sqrt(
            np.random.uniform(r_min**2, r_max**2)
        )

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([x, y, z], dtype=np.float32)

    def _update_target(self):
        p.resetBasePositionAndOrientation(
            self.target_body,
            self.target_pos.tolist(),
            [0, 0, 0, 1],
        )

    def _get_obs(self):
        q, qd = self._get_joint_state()
        ee_pos, _ = self._get_ee_pos_and_rot()
        return np.concatenate(
            [q, qd, ee_pos, self.target_pos]
        ).astype(np.float32)

    # ======================================================
    # Gym API
    # ======================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        q0 = np.clip(self.q_ref, self.joint_lower, self.joint_upper)
        for i, j in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, int(j), float(q0[i]), 0.0)

        self.target_pos = self._sample_target()
        self._update_target()

        for _ in range(10):
            p.stepSimulation()

        ee_pos, _ = self._get_ee_pos_and_rot()
        self.prev_dist = float(np.linalg.norm(ee_pos - self.target_pos))

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        q, qd = self._get_joint_state()

        delta_q = action * self.delta_scale
        q_target = np.clip(
            q + delta_q,
            self.joint_lower,
            self.joint_upper,
        )

        for i, j in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                int(j),
                p.POSITION_CONTROL,
                targetPosition=float(q_target[i]),
                positionGain=0.7,
                velocityGain=1.0,
                force=float(self.max_forces[i]),
            )

        for _ in range(self.n_substeps):
            p.stepSimulation()

        obs = self._get_obs()
        ee_pos, ee_rot = self._get_ee_pos_and_rot()

        dist = float(np.linalg.norm(ee_pos - self.target_pos))
        dist_improve = self.prev_dist - dist
        self.prev_dist = dist

        # ---------- reward ----------
        reward = 10.0 * dist_improve
        reward -= 0.01 * float(np.sum((q - self.q_ref) ** 2))
        reward -= 0.01 * float(np.sum(delta_q**2))

        to_target = self.target_pos - ee_pos
        if np.linalg.norm(to_target) > 1e-6:
            to_target /= np.linalg.norm(to_target)
            ee_z = ee_rot[:, 2]
            reward -= 0.1 * (1.0 - np.dot(ee_z, to_target))

        terminated = dist < self.distance_threshold
        truncated = self.step_count >= self.max_episode_steps

        if terminated:
            reward += 2.0

        info = {
            "distance": dist,
            "success": terminated,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass

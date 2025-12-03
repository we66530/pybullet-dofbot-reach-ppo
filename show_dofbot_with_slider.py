import pybullet as p
import time
import os
import math

# =========================
# Connect GUI
# =========================
p.connect(p.GUI)

# =========================
# Load DOFBOT URDF
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(
    BASE_DIR,
    "dofbot",
    "assets",
    "dofbot",
    "dofbot.urdf"
)

robot_id = p.loadURDF(
    URDF_PATH,
    basePosition=[0, 0, 0],
    useFixedBase=True
)

print("✅ DOFBOT loaded, robot_id =", robot_id)

# =========================
# Create joint sliders
# =========================
joint_sliders = []

num_joints = p.getNumJoints(robot_id)
print("Number of joints:", num_joints)

for j in range(num_joints):
    info = p.getJointInfo(robot_id, j)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]

    # only revolute joints
    if joint_type == p.JOINT_REVOLUTE:
        lower = info[8]
        upper = info[9]

        # fallback if no limits
        if lower > upper:
            lower = -math.pi
            upper = math.pi

        slider = p.addUserDebugParameter(
            joint_name,
            lower,
            upper,
            0.0
        )
        joint_sliders.append((j, slider))
        print(f"✅ Slider added: joint {j} ({joint_name})")

# =========================
# Control loop
# =========================
while p.isConnected():
    for joint_index, slider_id in joint_sliders:
        target_pos = p.readUserDebugParameter(slider_id)

        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=50
        )

    p.stepSimulation()
    time.sleep(1 / 240)
